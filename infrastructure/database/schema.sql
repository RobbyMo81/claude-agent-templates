-- Agent Army Database Schema v2.0
-- PostgreSQL 15+
-- Enhanced design with security, compliance, and cost tracking

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_cron";  -- For automated retention

-- ============================================================================
-- USERS & AUTHENTICATION
-- ============================================================================

CREATE TYPE user_role AS ENUM ('developer', 'approver', 'admin');

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    role user_role NOT NULL DEFAULT 'developer',
    monthly_budget_usd DECIMAL(10, 2) DEFAULT 100.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,  -- Soft delete for GDPR

    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_deleted ON users(deleted_at) WHERE deleted_at IS NOT NULL;

COMMENT ON TABLE users IS 'System users with RBAC roles';
COMMENT ON COLUMN users.deleted_at IS 'Soft delete timestamp for GDPR compliance';

-- ============================================================================
-- MISSIONS
-- ============================================================================

CREATE TYPE mission_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
CREATE TYPE retention_policy AS ENUM ('standard', 'extended', 'permanent');
CREATE TYPE environment_type AS ENUM ('dev', 'staging', 'production');

CREATE TABLE missions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),

    -- Specification (JSONB for flexibility)
    specification JSONB NOT NULL,
    environment environment_type NOT NULL DEFAULT 'dev',

    -- Workflow tracking
    workflow_id VARCHAR(255) UNIQUE,  -- Temporal workflow ID
    status mission_status NOT NULL DEFAULT 'pending',
    progress DECIMAL(3, 2) DEFAULT 0.00 CHECK (progress >= 0 AND progress <= 1),
    current_step VARCHAR(255),

    -- Results
    result JSONB,
    error_message TEXT,

    -- Cost tracking
    estimated_cost_usd DECIMAL(10, 4),
    actual_cost_usd DECIMAL(10, 4),
    llm_requests INTEGER DEFAULT 0,
    llm_cache_hits INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Data retention
    retention_policy retention_policy DEFAULT 'standard',
    auto_delete_at TIMESTAMP WITH TIME ZONE GENERATED ALWAYS AS (
        CASE retention_policy
            WHEN 'standard' THEN created_at + INTERVAL '90 days'
            WHEN 'extended' THEN created_at + INTERVAL '1 year'
            WHEN 'permanent' THEN NULL
        END
    ) STORED,

    -- Approval tracking (for production deployments)
    requires_approval BOOLEAN GENERATED ALWAYS AS (environment = 'production') STORED,
    approved_by UUID REFERENCES users(id),
    approved_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_approval CHECK (
        NOT requires_approval OR
        (approved_by IS NOT NULL AND approved_at IS NOT NULL) OR
        status IN ('pending', 'cancelled', 'failed')
    )
);

-- Indexes for performance
CREATE INDEX idx_missions_user_status ON missions(user_id, status);
CREATE INDEX idx_missions_status ON missions(status) WHERE status IN ('pending', 'running');
CREATE INDEX idx_missions_workflow ON missions(workflow_id);
CREATE INDEX idx_missions_created ON missions(created_at DESC);
CREATE INDEX idx_missions_auto_delete ON missions(auto_delete_at) WHERE auto_delete_at IS NOT NULL;
CREATE INDEX idx_missions_environment ON missions(environment);

-- JSONB indexes for specification queries
CREATE INDEX idx_missions_spec_action ON missions((specification->>'action'));

COMMENT ON TABLE missions IS 'User-submitted missions with workflow tracking';
COMMENT ON COLUMN missions.specification IS 'Mission configuration in JSONB format';
COMMENT ON COLUMN missions.workflow_id IS 'Temporal workflow ID for correlation';
COMMENT ON COLUMN missions.auto_delete_at IS 'Automatic deletion timestamp for GDPR compliance';

-- ============================================================================
-- AUDIT LOG (Immutable, Append-Only)
-- ============================================================================

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    -- Actor
    user_id UUID REFERENCES users(id),
    user_email VARCHAR(255),  -- Denormalized for historical record
    user_role user_role,

    -- Action
    action VARCHAR(100) NOT NULL,  -- e.g., 'mission.created', 'mission.approved', 'user.deleted'
    resource_type VARCHAR(50) NOT NULL,  -- e.g., 'mission', 'user'
    resource_id UUID,

    -- Details
    details JSONB,

    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id UUID,

    -- Compliance
    retention_until TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 years'),

    -- Immutability constraint: prevent updates/deletes
    CONSTRAINT audit_log_immutable CHECK (id > 0)
);

-- Indexes for audit queries
CREATE INDEX idx_audit_log_user ON audit_log(user_id, timestamp DESC);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_log_action ON audit_log(action, timestamp DESC);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_retention ON audit_log(retention_until) WHERE retention_until < NOW();

COMMENT ON TABLE audit_log IS 'Immutable audit trail for SOC2/GDPR compliance (7-year retention)';
COMMENT ON CONSTRAINT audit_log_immutable ON audit_log IS 'Ensures id is positive; immutability is enforced by rules below';

-- Prevent updates and deletes on audit_log
CREATE RULE audit_log_no_update AS ON UPDATE TO audit_log DO INSTEAD NOTHING;
CREATE RULE audit_log_no_delete AS ON DELETE TO audit_log DO INSTEAD NOTHING;

-- ============================================================================
-- COST TRACKING
-- ============================================================================

CREATE TABLE cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id UUID NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),

    -- Cost breakdown
    llm_provider VARCHAR(50),  -- 'claude', 'openai', 'azure', 'ollama'
    llm_model VARCHAR(50),     -- 'claude-sonnet-4.5', 'gpt-4', etc.
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd DECIMAL(10, 6),

    -- Cache metrics
    cache_hit BOOLEAN DEFAULT FALSE,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_cost_tracking_mission ON cost_tracking(mission_id);
CREATE INDEX idx_cost_tracking_user_month ON cost_tracking(user_id, created_at);
CREATE INDEX idx_cost_tracking_provider ON cost_tracking(llm_provider, created_at);

COMMENT ON TABLE cost_tracking IS 'Granular cost tracking for LLM API usage';

-- ============================================================================
-- LLM CACHE
-- ============================================================================

CREATE TABLE llm_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Cache key (content-addressable)
    prompt_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 of prompt
    prompt TEXT NOT NULL,

    -- Response
    response TEXT NOT NULL,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,

    -- Metadata
    input_tokens INTEGER,
    output_tokens INTEGER,
    generation_time_ms INTEGER,

    -- Cache management
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '30 days'),

    CONSTRAINT valid_hash CHECK (prompt_hash ~* '^[a-f0-9]{64}$')
);

CREATE INDEX idx_llm_cache_hash ON llm_cache(prompt_hash);
CREATE INDEX idx_llm_cache_expires ON llm_cache(expires_at) WHERE expires_at < NOW();
CREATE INDEX idx_llm_cache_access ON llm_cache(last_accessed_at);

COMMENT ON TABLE llm_cache IS 'Content-addressable cache for LLM responses (30-day TTL)';

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit log trigger for missions
CREATE OR REPLACE FUNCTION audit_mission_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, details)
        VALUES (NEW.user_id, 'mission.created', 'mission', NEW.id,
                jsonb_build_object('specification', NEW.specification, 'environment', NEW.environment));
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.status != NEW.status THEN
            INSERT INTO audit_log (user_id, action, resource_type, resource_id, details)
            VALUES (NEW.user_id, 'mission.status_changed', 'mission', NEW.id,
                    jsonb_build_object('old_status', OLD.status, 'new_status', NEW.status));
        END IF;

        IF NEW.approved_by IS NOT NULL AND OLD.approved_by IS NULL THEN
            INSERT INTO audit_log (user_id, action, resource_type, resource_id, details)
            VALUES (NEW.approved_by, 'mission.approved', 'mission', NEW.id,
                    jsonb_build_object('environment', NEW.environment));
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, details)
        VALUES (OLD.user_id, 'mission.deleted', 'mission', OLD.id, NULL);
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER missions_audit AFTER INSERT OR UPDATE OR DELETE ON missions
    FOR EACH ROW EXECUTE FUNCTION audit_mission_changes();

-- LLM cache access tracking
CREATE OR REPLACE FUNCTION update_cache_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed_at = NOW();
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER llm_cache_access BEFORE UPDATE ON llm_cache
    FOR EACH ROW EXECUTE FUNCTION update_cache_access();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- User budget view
CREATE OR REPLACE VIEW user_budgets AS
SELECT
    u.id,
    u.email,
    u.monthly_budget_usd,
    COALESCE(SUM(m.actual_cost_usd), 0) AS spent_this_month_usd,
    u.monthly_budget_usd - COALESCE(SUM(m.actual_cost_usd), 0) AS remaining_usd,
    COUNT(m.id) FILTER (WHERE m.status = 'completed') AS missions_completed_this_month,
    DATE_TRUNC('month', NOW()) AS period_start,
    DATE_TRUNC('month', NOW()) + INTERVAL '1 month' AS period_end
FROM users u
LEFT JOIN missions m ON u.id = m.user_id
    AND m.created_at >= DATE_TRUNC('month', NOW())
    AND m.status = 'completed'
WHERE u.deleted_at IS NULL
GROUP BY u.id, u.email, u.monthly_budget_usd;

COMMENT ON VIEW user_budgets IS 'Real-time view of user budget consumption';

-- Mission statistics view
CREATE OR REPLACE VIEW mission_stats AS
SELECT
    DATE_TRUNC('day', created_at) AS date,
    environment,
    status,
    COUNT(*) AS count,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS avg_duration_seconds,
    AVG(actual_cost_usd) AS avg_cost_usd,
    SUM(actual_cost_usd) AS total_cost_usd
FROM missions
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at), environment, status
ORDER BY date DESC;

COMMENT ON VIEW mission_stats IS 'Daily mission statistics for dashboards';

-- ============================================================================
-- AUTOMATED MAINTENANCE (pg_cron)
-- ============================================================================

-- Daily: Delete expired missions
SELECT cron.schedule(
    'delete-expired-missions',
    '0 2 * * *',  -- 2 AM daily
    $$DELETE FROM missions WHERE auto_delete_at < NOW()$$
);

-- Daily: Delete expired LLM cache entries
SELECT cron.schedule(
    'delete-expired-cache',
    '0 3 * * *',  -- 3 AM daily
    $$DELETE FROM llm_cache WHERE expires_at < NOW()$$
);

-- Weekly: Vacuum and analyze for performance
SELECT cron.schedule(
    'vacuum-analyze',
    '0 4 * * 0',  -- 4 AM Sunday
    $$VACUUM ANALYZE$$
);

-- Monthly: Archive old audit logs (export to Cloud Storage before delete)
-- Note: Audit logs have 7-year retention, this is just for very old logs

-- Create a procedure to archive and delete old audit logs
CREATE OR REPLACE PROCEDURE archive_audit_logs()
LANGUAGE plpgsql
AS $$
DECLARE
    archive_filename TEXT;
    copy_command TEXT;
BEGIN
    -- Construct the filename with current year and month
    archive_filename := 'gs://agent-army-archive/audit-logs-' || TO_CHAR(NOW(), 'YYYYMM') || '.csv.gz';
    copy_command := 'gzip | gsutil cp - ' || archive_filename;

    -- Export logs older than 7 years to external storage
    EXECUTE format(
        $f$COPY (
            SELECT * FROM audit_log WHERE retention_until < NOW()
        ) TO PROGRAM %L$f$,
        copy_command
    );

    -- Then delete
    DELETE FROM audit_log WHERE retention_until < NOW();
END;
$$;

-- Schedule the procedure to run monthly
SELECT cron.schedule(
    'archive-audit-logs',
    '0 1 1 * *',  -- 1 AM on 1st of month
    $$CALL archive_audit_logs();$$
);

-- ============================================================================
-- SEED DATA (Development Only)
-- ============================================================================

-- Create admin user
INSERT INTO users (email, role, monthly_budget_usd)
VALUES ('admin@agent-army.dev', 'admin', 1000.00)
ON CONFLICT (email) DO NOTHING;

-- Create test developer user
INSERT INTO users (email, role, monthly_budget_usd)
VALUES ('developer@agent-army.dev', 'developer', 100.00)
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- SECURITY
-- ============================================================================

-- Row-level security (RLS) for multi-tenancy
ALTER TABLE missions ENABLE ROW LEVEL SECURITY;

-- Users can only see their own missions (unless admin)
CREATE POLICY missions_isolation ON missions
    FOR ALL
    USING (
        user_id = current_setting('app.user_id', TRUE)::UUID
        OR
        (SELECT role FROM users WHERE id = current_setting('app.user_id', TRUE)::UUID) = 'admin'
    );

COMMENT ON POLICY missions_isolation ON missions IS 'RLS: Users can only access their own missions (admins see all)';

-- Grant permissions (application user)
-- CREATE ROLE agent_army_app WITH LOGIN PASSWORD 'change_me_in_production';
-- GRANT SELECT, INSERT, UPDATE ON users, missions, audit_log, cost_tracking, llm_cache TO agent_army_app;
-- GRANT SELECT ON user_budgets, mission_stats TO agent_army_app;
-- GRANT USAGE ON SEQUENCE audit_log_id_seq TO agent_army_app;

-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

CREATE TABLE schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES ('2.0.0', 'Enhanced design with security, compliance, and cost tracking');

-- End of schema
