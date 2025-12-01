import React, { useState, useEffect } from 'react';

interface ModelNameInputProps {
    value: string;
    onChange: (name: string) => void;
    existingModels: string[];
    suggestions?: string[];
    placeholder?: string;
    required?: boolean;
}

export const ModelNameInput: React.FC<ModelNameInputProps> = ({
    value,
    onChange,
    existingModels,
    suggestions = [],
    placeholder = "Enter model name...",
    required = true
}) => {
    const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
    const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
    const [isDuplicate, setIsDuplicate] = useState<boolean>(false);
    const [validationMessage, setValidationMessage] = useState<string>('');

    // Default suggestions based on common patterns
    const defaultSuggestions = [
        'agent_v1',
        'agent_v2',
        'powerball_predictor',
        'lottery_agent',
        'neural_predictor',
        'ml_agent'
    ];

    const allSuggestions = [...suggestions, ...defaultSuggestions];

    useEffect(() => {
        validateModelName(value);
    }, [value, existingModels]);

    const validateModelName = (name: string) => {
        if (!name.trim()) {
            setValidationMessage('');
            setIsDuplicate(false);
            return;
        }

        // Check for duplicates
        if (existingModels.includes(name)) {
            setIsDuplicate(true);
            setValidationMessage('This model name already exists');
            return;
        }

        // Check naming conventions
        if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
            setValidationMessage('Model name can only contain letters, numbers, underscores, and hyphens');
            setIsDuplicate(false);
            return;
        }

        if (name.length < 3) {
            setValidationMessage('Model name must be at least 3 characters long');
            setIsDuplicate(false);
            return;
        }

        if (name.length > 50) {
            setValidationMessage('Model name must be less than 50 characters');
            setIsDuplicate(false);
            return;
        }

        // Valid name
        setValidationMessage('');
        setIsDuplicate(false);
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        onChange(newValue);

        // Filter suggestions based on input
        if (newValue.trim()) {
            const filtered = allSuggestions.filter(suggestion =>
                suggestion.toLowerCase().includes(newValue.toLowerCase()) &&
                !existingModels.includes(suggestion)
            );
            setFilteredSuggestions(filtered);
            setShowSuggestions(filtered.length > 0);
        } else {
            setShowSuggestions(false);
        }
    };

    const handleSuggestionClick = (suggestion: string) => {
        onChange(suggestion);
        setShowSuggestions(false);
    };

    const handleFocus = () => {
        if (value.trim() && filteredSuggestions.length > 0) {
            setShowSuggestions(true);
        }
    };

    const handleBlur = () => {
        // Delay hiding suggestions to allow for clicks
        setTimeout(() => setShowSuggestions(false), 150);
    };

    const generateSuggestion = () => {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_');
        const suggestion = `agent_${timestamp}`;
        onChange(suggestion);
    };

    const isValid = !isDuplicate && !validationMessage && value.trim().length >= 3;

    return (
        <div className="relative">
            <div className="flex items-center space-x-2">
                <div className="flex-1">
                    <label htmlFor="modelName" className="block text-sm font-medium text-gray-300 mb-1">
                        Model Name
                    </label>
                    <input
                        id="modelName"
                        type="text"
                        value={value}
                        onChange={handleInputChange}
                        onFocus={handleFocus}
                        onBlur={handleBlur}
                        placeholder={placeholder}
                        required={required}
                        className={`block w-full bg-gray-700 border rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-2 focus:border-transparent sm:text-sm ${
                            isDuplicate
                                ? 'border-red-500 focus:ring-red-500'
                                : validationMessage
                                    ? 'border-yellow-500 focus:ring-yellow-500'
                                    : isValid
                                        ? 'border-green-500 focus:ring-green-500'
                                        : 'border-gray-600 focus:ring-blue-500'
                        }`}
                    />
                </div>
                <button
                    type="button"
                    onClick={generateSuggestion}
                    className="mt-6 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-gray-300 text-sm rounded transition-colors"
                    title="Generate unique name"
                >
                    ✨
                </button>
            </div>

            {/* Validation Message */}
            {validationMessage && (
                <div className={`mt-1 text-xs ${isDuplicate ? 'text-red-400' : 'text-yellow-400'}`}>
                    {validationMessage}
                </div>
            )}

            {/* Success Indicator */}
            {isValid && value.trim() && (
                <div className="mt-1 text-xs text-green-400">
                    ✓ Valid model name
                </div>
            )}

            {/* Suggestions Dropdown */}
            {showSuggestions && filteredSuggestions.length > 0 && (
                <div className="absolute z-10 mt-1 w-full bg-gray-700 border border-gray-600 rounded-md shadow-lg max-h-60 overflow-auto">
                    <div className="py-1">
                        <div className="px-3 py-1 text-xs text-gray-400 border-b border-gray-600">
                            Suggestions
                        </div>
                        {filteredSuggestions.slice(0, 5).map((suggestion, index) => (
                            <button
                                key={index}
                                type="button"
                                onClick={() => handleSuggestionClick(suggestion)}
                                className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-600 focus:bg-gray-600 focus:outline-none"
                            >
                                {suggestion}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
