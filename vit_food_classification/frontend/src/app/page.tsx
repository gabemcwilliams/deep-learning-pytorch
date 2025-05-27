'use client';  // Enables client-side interactivity (required in Next.js app dir structure)

import {useState} from 'react';

// Main functional component for the page
export default function Home() {
    // File input state
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);

    // Prediction result and error state
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    // Handle image selection
    function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setImagePreviewUrl(URL.createObjectURL(file));  // Generate preview URL
            setResult(null);
            setError(null);
        }
    }

    // Handle image submission to FastAPI backend
    async function handleSubmit() {
        if (!selectedFile) {
            setError("Please select an image.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("http://backend-server:8000/api/v1/upload/image", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error ${response.status}`);
            }

            const data = await response.json();

            // Flatten results into class/score pairs
            const probs = data.prediction.probs;
            const labels = Object.keys(probs);
            const values = labels.map(label => probs[label]);

            setResult({
                class_name: data.prediction.predicted_label,
                class_labels: labels,
                pred_probs: [values],  // Stored as 2D for potential chart use
            });
        } catch (err) {
            console.error("Prediction error:", err);
            setError("Prediction failed. Check console for details.");
        }
    }

    // UI Rendering
    return (
        <main className="min-h-screen p-8 flex flex-col items-center">
            <h1 className="text-4xl font-bold mb-6">Steak, Pizza, or Sushi?</h1>

            <div className="flex flex-col sm:flex-row gap-8 w-full max-w-4xl">
                {/* Upload and results */}
                <div className="flex flex-col gap-4 flex-1">
                    <input type="file" accept="image/*" onChange={handleFileChange}/>

                    <button
                        onClick={handleSubmit}
                        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
                    >
                        Submit
                    </button>

                    {error && <p className="text-red-600">{error}</p>}

                    {result && (
                        <div>
                            <h2 className="text-xl font-semibold mt-4">Prediction:</h2>
                            <p className="text-green-700 text-lg">{result.class_name}</p>
                            <ul className="mt-2">
                                {result.class_labels.map((label: string, i: number) => (
                                    <li key={label}>
                                        {label}: {(result.pred_probs[0][i] * 100).toFixed(2)}%
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>

                {/* Image preview */}
                <div className="flex-1 border border-gray-300 rounded p-4">
                    {imagePreviewUrl ? (
                        <img
                            src={imagePreviewUrl}
                            alt="Preview"
                            className="w-full max-h-96 object-contain"
                        />
                    ) : (
                        <p className="text-gray-500">Image preview will appear here.</p>
                    )}
                </div>
            </div>
        </main>
    );
}
