import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Nav from './components/nav/nav';
import Hero from './components/Hero/hero';
import HDBResellPage from './forecast/Page';
import HDBHelperPage from './helper/Page';
import HDBMarketPage from './marketplace/Page';

function App() {
  return (
    <Router>
  <div className="relative w-full h-screen overflow-hidden">
    <div className="absolute inset-0 z-0">
      <Nav />
    </div>

    <div className="relative z-10 mt-[8vh]">
      <Routes>
        <Route path="/" element={<Hero />} />
        <Route path="/forecast" element={<HDBResellPage />} />
        <Route path="/helper" element={<HDBHelperPage />} />
        <Route path="/marketplace" element={<HDBMarketPage />} />
      </Routes>
    </div>
  </div>
</Router>

  );
}

export default App;
