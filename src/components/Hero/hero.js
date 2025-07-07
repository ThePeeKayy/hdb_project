import React from 'react';

export default function Hero() {
  return (
    <div className="mx-auto max-w-4xl lg:max-w-7xl px-6 lg:px-8 bg-transparent mt-[20vh]">
      <div className="mx-auto max-w-8xl">
        <div className="text-center">
          <h1 className="py-4 text-balance text-5xl font-black tracking-tight bg-gradient-to-br from-white via-gray-300 to-black bg-clip-text text-transparent sm:text-7xl">
            HDB Resale Hub
          </h1>
          <div className='px-[100px]'>
            <p className="mt-8 text-pretty text-lg font-medium text-white sm:text-xl/8">
              Important Disclaimer: This is a demonstration project and this platform has no license. Prediction model is a fully functional prototype but AI advice is not federated by any government entities.
            </p>
          </div>
          <div className='flex flex-row justify-center gap-x-8 mt-4'>
            <a href="/forecast" className='hover:bg-black font-semibold min-w-[160px] sm:min-w-[200px] transform transition-all duration-200 hover:scale-105 hover:-translate-y-1 gap-y-2 flex items-center flex-col hover:text-white bg-white rounded-xl border-[2px] border-gray-700 p-8 sm:p-10'>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="size-10">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5m.75-9 3-3 2.148 2.148A12.061 12.061 0 0 1 16.5 7.605" />
              </svg>
              Forecast
            </a>
            <a href="/helper" className='hover:bg-black font-semibold min-w-[160px] sm:min-w-[200px] transform transition-all duration-200 hover:scale-105 hover:-translate-y-1 gap-y-2 flex items-center flex-col hover:text-white bg-white rounded-xl border-[2px] border-gray-700 p-8 sm:p-10'>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="size-10">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
              </svg>
              Seek Advice
            </a>
            <a href="/marketplace" className='hover:bg-black font-semibold min-w-[160px] sm:min-w-[200px] transform transition-all duration-200 hover:scale-105 hover:-translate-y-1 gap-y-2 flex items-center flex-col hover:text-white bg-white rounded-xl border-[2px] border-gray-700 p-8 sm:p-10'>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="size-10">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5V6a3.75 3.75 0 1 0-7.5 0v4.5m11.356-1.993 1.263 12c.07.665-.45 1.243-1.119 1.243H4.25a1.125 1.125 0 0 1-1.12-1.243l1.264-12A1.125 1.125 0 0 1 5.513 7.5h12.974c.576 0 1.059.435 1.119 1.007ZM8.625 10.5a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm7.5 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z" />
              </svg>
              Marketplace
            </a>
          </div>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <p className=' text-gray-300'>Data from <a href="https://data.gov.sg/" className='font-bold text-purple-500'>data.gov.sg</a></p>
          </div>
        </div>
      </div>
    </div>
  );
}
