import React, { useState } from 'react';

const HDBHelperPage = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const API_URL = process.env.REACT_APP_API_URL;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setResponse('');

    try {
      const res = await fetch(`${API_URL}/api/helper`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim() })
      });

      const data = await res.json();
      setResponse(data.advice);
    } catch (err) {
      setResponse('Failed to get response. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">HDB Resale AI Helper</h1>
          <p className="text-gray-400">Trained based on regulations data</p>
        </div>

        <div className="bg-gray-800 rounded-lg shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Question:
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="(Max 300 characters)"
                className="w-full h-32 p-4 bg-gray-700 border border-gray-100 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                maxLength={300}
                disabled={loading}
              />
            </div>

            <div className="flex gap-4">
              <button
                type="submit"
                disabled={loading || !prompt.trim()}
                className="flex-1 bg-gray-800 border-gray-100 border text-white py-3 px-6 rounded-lg font-medium hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Processing...' : 'Get Answer'}
              </button>

            </div>
          </form>

          {response && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-white mb-3">Response:</h3>
              <div className="bg-gray-700 border border-gray-100 rounded-lg max-h-[30vh] overflow-auto p-6">
                <p className="text-gray-200 leading-relaxed">{response}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HDBHelperPage;