
import React from 'react';

const Spinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center space-y-4">
       <div className="w-16 h-16 border-4 border-blue-400 border-t-transparent border-solid rounded-full animate-spin"></div>
       <p className="text-lg text-blue-300">Running Analysis...</p>
    </div>
  );
};

export default Spinner;
