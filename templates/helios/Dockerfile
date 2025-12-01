# Frontend Dockerfile
FROM node:22.13.1-alpine3.21 as builder

# Set working directory
WORKDIR /app

# Install security updates
RUN apk update && apk upgrade

# Copy package files
COPY package*.json ./

# Install ALL dependencies (including vite and other build tools)
RUN npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Install security updates
RUN apk update && apk upgrade

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Remove the custom nginx configuration line since we'll modify it dynamically
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001 -G nodejs

# Expose port (Cloud Run will override this with PORT env var)
EXPOSE 8080

# Health check - update to use $PORT variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT:-8080}/ || exit 1

# Dynamic port configuration and start nginx
# This replaces the default port 80 with the PORT environment variable provided by Cloud Run
CMD sed -i -e "s/80/${PORT:-8080}/g" /etc/nginx/conf.d/default.conf && nginx -g "daemon off;"
