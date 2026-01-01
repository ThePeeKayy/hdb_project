export default function Hero() {
  return (
    <div className="relative min-h-screen flex items-end pb-[130px] px-6 mr-4 lg:px-12">
      <div className="absolute top-0 bg-blackleft-6 lg:left-12 max-w-md">
        <p className="text-white/90 text-lg leading-relaxed backdrop-blur-sm  p-6 rounded-lg border border-white/10">
          Discover HDB resale prices with AI-powered forecasting at the intersection of data and community insights.
        </p>
      </div>

      <div className="absolute lg:bottom-[150px] bottom-[52%] flex-1 ">
        <h1 className="text-[45px] xl:text-[7rem] font-black tracking-tighter text-white leading-none">
          HDB
          <br />
          RESALE
          <br />
          HUB
        </h1>
      </div>

      <div className="hidden lg:absolute right-[20px] lg:grid grid-cols-2 gap-4 w-full max-w-2xl">
        {/* Featured card - Forecast (spans 2 columns) */}
        <a
          href="/forecast"
          className="col-span-2 group relative overflow-hidden bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-md rounded-2xl p-8 border border-white/20 hover:border-cyan-400 transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:shadow-cyan-500/30"
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                Price Forecasting
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="2"
                  stroke="currentColor"
                  className="size-6 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 19.5 15-15m0 0H8.25m11.25 0v11.25" />
                </svg>
              </h3>
              <p className="text-white/80 text-sm leading-relaxed">
                AI-powered predictions for HDB resale prices across Singapore neighborhoods.
              </p>
            </div>
            <div className="ml-4 bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth="1.5"
                stroke="currentColor"
                className="size-12 text-cyan-300"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5m.75-9 3-3 2.148 2.148A12.061 12.061 0 0 1 16.5 7.605"
                />
              </svg>
            </div>
          </div>
        </a>

        {/* Seek Advice button */}
        <a
          href="/helper"
          className="group relative overflow-hidden bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all duration-300 hover:scale-[1.02] flex items-center justify-center"
        >
          <div className="text-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              className="size-10 text-white mx-auto mb-2"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z"
              />
            </svg>
            <span className="text-white font-semibold text-lg">Seek Advice</span>
          </div>
        </a>

        {/* Marketplace button */}
        <a
          href="/marketplace"
          className="group relative overflow-hidden bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all duration-300 hover:scale-[1.02] flex items-center justify-center"
        >
          <div className="text-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              className="size-10 text-white mx-auto mb-2"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5V6a3.75 3.75 0 1 0-7.5 0v4.5m11.356-1.993 1.263 12c.07.665-.45 1.243-1.119 1.243H4.25a1.125 1.125 0 0 1-1.12-1.243l1.264-12A1.125 1.125 0 0 1 5.513 7.5h12.974c.576 0 1.059.435 1.119 1.007ZM8.625 10.5a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm7.5 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
              />
            </svg>
            <span className="text-white font-semibold text-lg">Marketplace</span>
          </div>
        </a>

        {/* Data source attribution */}
        <div className="col-span-2 bg-black/40 backdrop-blur-sm rounded-2xl p-4 border border-white/10 text-center">
          <p className="text-white/70 text-sm">
            Data from{" "}
            <a href="https://data.gov.sg/" className="font-bold text-cyan-400 hover:text-cyan-300 transition-colors">
              data.gov.sg
            </a>
          </p>
        </div>
      </div>

      <div className="lg:hidden flex flex-col gap-3 w-full mt-8">
        <a
          href="/forecast"
          className="group bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-md rounded-xl p-6 border border-white/20 hover:border-cyan-400 transition-all"
        >
          <div className="flex items-center gap-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              className="size-8 text-cyan-300"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5m.75-9 3-3 2.148 2.148A12.061 12.061 0 0 1 16.5 7.605"
              />
            </svg>
            <span className="text-white font-bold text-lg">Forecast</span>
          </div>
        </a>

        <div className="grid grid-cols-2 gap-3">
          <a href="/helper" className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 text-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              className="size-8 text-white mx-auto mb-2"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z"
              />
            </svg>
            <span className="text-white font-semibold text-sm">Seek Advice</span>
          </a>

          <a
            href="/marketplace"
            className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 text-center"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              className="size-8 text-white mx-auto mb-2"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5V6a3.75 3.75 0 1 0-7.5 0v4.5m11.356-1.993 1.263 12c.07.665-.45 1.243-1.119 1.243H4.25a1.125 1.125 0 0 1-1.12-1.243l1.264-12A1.125 1.125 0 0 1 5.513 7.5h12.974c.576 0 1.059.435 1.119 1.007ZM8.625 10.5a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm7.5 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
              />
            </svg>
            <span className="text-white font-semibold text-sm">Marketplace</span>
          </a>
        </div>

        <div className="bg-black/40 backdrop-blur-sm rounded-xl p-3 border border-white/10 text-center">
          <p className="text-white/70 text-xs">
            Data from{" "}
            <a href="https://data.gov.sg/" className="font-bold text-cyan-400">
              data.gov.sg
            </a>
          </p>
        </div>
      </div>

      <div className="absolute bottom-4 left-6 right-6 lg:left-12 lg:right-12">
        <p className="text-white/50 text-xs leading-relaxed text-center lg:text-left">
          Important: Demonstration project. Prediction model is a prototype. AI advice is not endorsed by government
          entities.
        </p>
      </div>
    </div>
  )
}
