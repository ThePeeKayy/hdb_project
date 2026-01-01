import React, { useState } from 'react';

interface PredictionData {
  current_avg_price: number;
  predicted_6m_price: number;
  predicted_12m_price: number;
  trend: string;
}

export default function HDBResellPage() {
  const [selectedTown, setSelectedTown] = useState('');
  const [selectedFlatType, setSelectedFlatType] = useState('');
  const [selectedLeaseBucket, setSelectedLeaseBucket] = useState('');
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const towns = [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
    'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG',
    'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
    'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON',
    'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
  ];

  const flatTypes = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'];

  const leaseBuckets = [
    '0-20', '20-40', '40-60', '60-80'
  ];

  const handlePredict = async () => {
    if (!selectedTown || !selectedFlatType || !selectedLeaseBucket) {
      setError('Please select town, flat type, and lease remaining');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('https://vqe2yhjppn.ap-southeast-1.awsapprunner.com/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          town: selectedTown,
          flat_type: selectedFlatType,
          remaining_lease_bucket: selectedLeaseBucket,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const data = await response.json();
      setPredictionData(data);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-SG', {
      style: 'currency',
      currency: 'SGD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  return (
    <div className="min-h-screen py-4 flex justify-center">
      <div className='flex lg:flex-row flex-col w-full lg:justify-center justify-start lg:items-start items-center max-w-7xl px-4 gap-8'>
        
      <div className="max-w-2xl px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-50 mb-2">
            HDB Resale Price Predictor
          </h1>
          <p className="text-gray-200">
            Get price predictions for HDB resale flats
          </p>
        </div>

        <div className="bg-gray-800 rounded-lg shadow-md p-6 mb-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-200 mb-2">
                Town
              </label>
              <select
                value={selectedTown}
                onChange={(e) => setSelectedTown(e.target.value)}
                className="w-full bg-gray-50 px-3 py-2 border border-gray-300 rounded-md"
              >
                <option value="">Select a town</option>
                {towns.map((town) => (
                  <option key={town} value={town}>
                    {town}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-200 mb-2">
                Flat Type
              </label>
              <select
                value={selectedFlatType}
                onChange={(e) => setSelectedFlatType(e.target.value)}
                className="bg-gray-50 w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                <option value="">Select flat type</option>
                {flatTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-200 mb-2">
                Remaining Lease (Years)
              </label>
              <select
                value={selectedLeaseBucket}
                onChange={(e) => setSelectedLeaseBucket(e.target.value)}
                className="bg-gray-50 w-full px-3 py-2 border border-gray-300 rounded-md "
              >
                <option value="">Select lease remaining</option>
                {leaseBuckets.map((bucket) => (
                  <option key={bucket} value={bucket}>
                    {bucket} years
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-900 hover:border-white text-white rounded-md hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {loading ? 'Getting Prediction...' : 'Get Price Prediction'}
            </button>

            {error && (
              <div className="p-3 bg-red-100 border border-red-300 rounded-md text-red-700 text-sm">
                {error}
              </div>
            )}
          </div>
        </div>

        {predictionData && (
          <div className="bg-gray-800 rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-white mb-4">Price Predictions</h2>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b">
                <span className="text-gray-600">Current Average Price</span>
                <span className="font-semibold text-lg text-white">
                  {formatPrice(predictionData.current_avg_price)}
                </span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b">
                <span className="text-gray-600">6-Month Prediction</span>
                <span className="font-semibold text-lg text-green-600">
                  {formatPrice(predictionData.predicted_6m_price)}
                </span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b">
                <span className="text-gray-600">12-Month Prediction</span>
                <span className="font-semibold text-lg text-blue-600">
                  {formatPrice(predictionData.predicted_12m_price)}
                </span>
              </div>
              
              <div className="flex justify-between items-center py-2">
                <span className="text-gray-600">Market Trend</span>
                <span className={`font-semibold capitalize ${
                  predictionData.trend === 'increasing' ? 'text-green-600' :
                  predictionData.trend === 'decreasing' ? 'text-red-600' :
                  'text-gray-600'
                }`}>
                  {predictionData.trend === 'increasing' && '📈 '}
                  {predictionData.trend === 'decreasing' && '📉 '}
                  {predictionData.trend === 'stable' && '➡️ '}
                  {predictionData.trend === 'No sale' && '🚫'}
                  {predictionData.trend}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="max-w-2xl flex-1">
        <div className="flex flex-col justify-center text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-50 mb-2">
            Live HDB Data
          </h1>
          <p className="text-gray-200">
            Official numbers from gov data
          </p>
              <iframe
                className="mt-8 sm:block hidden bg-gray-800 rounded-lg scale-75 md:scale-100 origin-top-left"
                width="600"
                height="600"
                src="https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/chart/1421"
                title="Live HDB Data"
              />
              <iframe
                className="mt-8 ml-[70px] block sm:hidden mx-auto bg-gray-800 rounded-lg scale-75 origin-top-left"
                width="400"
                height="400"
                src="https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/chart/1421"
                title="Live HDB Data"
              />

        </div>
      </div>
      
      </div>

    </div>
  );
}