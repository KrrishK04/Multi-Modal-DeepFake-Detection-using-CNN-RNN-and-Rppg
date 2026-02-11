
import React, { useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileUploaderProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
}

const FileUploader = ({ onFileSelect, selectedFile }: FileUploaderProps) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    } else {
      onFileSelect(null);
    }
  };
  
  return (
    <div className="w-full max-w-lg">
      <label
        htmlFor="file-upload"
        className={cn(
          "relative flex flex-col items-center justify-center w-full h-64 glassmorphism cursor-pointer transition-all duration-300",
          isDragOver ? "border-cyan-400 shadow-cyan-500/20" : "border-white/10",
          "hover:border-cyan-400 hover:shadow-cyan-500/20"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
          <UploadCloud className="w-10 h-10 mb-4 text-gray-400" />
          <p className="mb-2 text-sm text-gray-300">
            <span className="font-semibold text-cyan-400">Drag & Drop a Video Here</span> or Click to Upload
          </p>
        </div>
        <input id="file-upload" type="file" className="hidden" accept=".mp4,.mov,.avi" onChange={handleFileChange} />
      </label>
      {selectedFile && (
        <div className="mt-4 text-sm text-gray-400 text-center">
          Selected: <span className="font-medium text-gray-200">{selectedFile.name}</span>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
