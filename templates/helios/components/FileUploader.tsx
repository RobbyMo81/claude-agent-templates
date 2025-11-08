
import React, { useRef } from 'react';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileSelect, selectedFile }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      onFileSelect(event.target.files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".csv"
      />
      <button
        onClick={handleClick}
        className="w-full bg-gray-700 hover:bg-gray-600 text-gray-300 border border-dashed border-gray-500 rounded-lg p-4 text-center cursor-pointer transition-colors"
      >
        <div className="flex flex-col items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 mb-2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span className="font-semibold">
                {selectedFile ? 'Change File' : 'Upload CSV'}
            </span>
            {selectedFile && (
                 <span className="text-xs text-gray-400 mt-1 truncate max-w-full">{selectedFile.name}</span>
            )}
            {!selectedFile && (
                 <span className="text-xs text-gray-400 mt-1">Powerball history data</span>
            )}
        </div>
      </button>
    </div>
  );
};

export default FileUploader;
