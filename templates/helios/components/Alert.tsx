
import React from 'react';

interface AlertProps {
  message: string;
  type?: 'error' | 'warning' | 'info';
}

const Alert: React.FC<AlertProps> = ({ message, type = 'error' }) => {
  const baseClasses = 'p-4 border-l-4 rounded-md';
  const typeClasses = {
    error: 'bg-red-900/50 border-red-500 text-red-200',
    warning: 'bg-yellow-900/50 border-yellow-500 text-yellow-200',
    info: 'bg-blue-900/50 border-blue-500 text-blue-200',
  };

  return (
    <div className={`${baseClasses} ${typeClasses[type]}`} role="alert">
      <p className="font-bold capitalize">{type}</p>
      <p>{message}</p>
    </div>
  );
};

export default Alert;
