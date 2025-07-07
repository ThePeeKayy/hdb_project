import React, { useState } from "react";

export default function AddFlatForm({ setForm }) {
  const API_URL = process.env.REACT_APP_API_URL;
  const [formDetails, setFormDetails] = useState({
    flatType: "",
    location: "",
    title: "",
    description: "",
    name: "",
    telegram: "",
    image: "",
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const areas = [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
    'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG',
    'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
    'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON',
    'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
  ];

  const flatTypes = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'];

  const getCurrentFormattedDate = () => {
    const now = new Date();
    const options = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    };
    return now.toLocaleDateString('en-US', options);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormDetails((prev) => ({ ...prev, [name]: value }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
      }
      
      if (file.size > 1024 * 1024) {
        alert('Image size should be less than 1MB');
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const base64String = event.target.result;
        setFormDetails((prev) => ({ ...prev, image: base64String }));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    if (!formDetails.flatType || !formDetails.location || !formDetails.title || 
        !formDetails.description || !formDetails.name || !formDetails.telegram) {
      alert('Please fill in all required fields');
      setIsSubmitting(false);
      return;
    
    }

    const flatData = {
      title: formDetails.title,
      description: formDetails.description,
      imageUrl: formDetails.image || '',
      date: getCurrentFormattedDate(),
      datetime: new Date().toISOString().split('T')[0], 
      category: formDetails.flatType, 
      location: formDetails.location,
      author: {
        name: formDetails.name,
        telegram: formDetails.telegram.startsWith('@') ? formDetails.telegram : `@${formDetails.telegram}`,
      },
    };
    try {
      const response = await fetch(`${API_URL}/api/listings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(flatData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error submitting flat:', errorData);
        alert('Error submitting flat listing. Please try again.');
      }
      setForm(false);
    } catch (error) {
      console.error('Network error:', error);
      alert('Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="px-6">
      <div className="relative bg-white mx-auto max-w-3xl rounded-lg max-h-[85vh] overflow-y-auto scrollbar-hide">
        <section className="py-16">
          <form onSubmit={handleSubmit} className="space-y-6 px-8">
            <div className="flex justify-between items-center">
              <h3 className="text-xl font-medium text-gray-900">Flat Details</h3>
              <button className='cursor-pointer' type="button" onClick={() => setForm(false)} aria-label="Close">
                ✕
              </button>
            </div>
            
            <div className="flex flex-row justify-between">
              <div>
                <label className="min-w-[35vw] md:min-w-[15vw] block text-sm font-medium text-gray-700">
                  Flat Type *
                </label>
                <select
                  name="flatType"
                  value={formDetails.flatType}
                  onChange={handleChange}
                  required
                  className="block w-full rounded-md mt-2 border-[2px] p-3 border-gray-300"
                >
                  <option value="">Select flat type</option>
                  {flatTypes.map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Area *
                </label>
                <select
                  name="location"
                  value={formDetails.location}
                  onChange={handleChange}
                  required
                  className="min-w-[35vw] md:min-w-[15vw] block w-full rounded-md border-[2px] p-3 mt-2 border-gray-300"
                >
                  <option value="">Select area</option>
                  {areas.map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Title *
              </label>
              <input
                type="text"
                name="title"
                value={formDetails.title}
                onChange={handleChange}
                required
                className="block w-full rounded-md border-[2px] p-3 mt-2 border-gray-300"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Description *
              </label>
              <textarea
                name="description"
                value={formDetails.description}
                onChange={handleChange}
                required
                rows={4}
                className="border-[2px] p-3 mt-2 block w-full rounded-md border-gray-300"
              />
            </div>

            {/* Image Upload Field */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Flat Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="border-[2px] p-3 mt-2 block w-full rounded-md border-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200"
              />
              {formDetails.image && (
                <div className="mt-3">
                  <img 
                    src={formDetails.image} 
                    alt="Preview" 
                    className="max-w-xs max-h-48 rounded-md border border-gray-300"
                  />
                </div>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Name *
              </label>
              <input
                type="text"
                name="name"
                value={formDetails.name}
                onChange={handleChange}
                required
                className="border-[2px] p-3 mt-2 block w-full rounded-md border-gray-300"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Telegram *
              </label>
              <input
                type="text"
                name="telegram"
                value={formDetails.telegram}
                onChange={handleChange}
                required
                placeholder="@username"
                className="border-[2px] p-3 mt-2 block w-full rounded-md border-gray-300"
              />
            </div>

            <button 
              type="submit" 
              disabled={isSubmitting}
              className="px-4 py-2 bg-gray-800 hover:bg-white hover:text-black border-[2px] mt-2 border-black hover:border-gray-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}