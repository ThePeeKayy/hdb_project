"use client"

import { useState, useEffect } from "react"
import {
  TrendingUp,
  Filter,
  X,
  Activity,
  Database,
  Layers,
  Sparkles,
  ArrowRight,
  Clock,
  ChevronDown,
} from "lucide-react"

interface PredictionData {
  region: string
  flat_type: string
  group_key: string
  current_avg_price: number
  predicted_1m_price: number
  predicted_2m_price: number
  predicted_3m_price: number
  trend: string
  confidence_score?: number
  last_updated?: string
}

interface AllPredictionsResponse {
  total_count: number
  predictions: PredictionData[]
  regions: string[]
  flat_types: string[]
  last_updated: string
}

interface ModelMetrics {
  timestamp: string
  model_type: string
  num_models: number
  test_performance?: {
    avg_rmse: number
    avg_mae: number
    avg_mape: number
  }
}

export default function HDBResellPage() {
  const [allData, setAllData] = useState<AllPredictionsResponse | null>(null)
  const [filteredData, setFilteredData] = useState<PredictionData[]>([])
  const [selectedRegion, setSelectedRegion] = useState("ALL")
  const [selectedFlatType, setSelectedFlatType] = useState("ALL")
  const [selectedTrend, setSelectedTrend] = useState("ALL")
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [showMetrics, setShowMetrics] = useState(false)
  const [expandedPipeline, setExpandedPipeline] = useState(false)

  const API_ENDPOINT = "https://dio4dmuffh.execute-api.ap-southeast-1.amazonaws.com/prod"

  useEffect(() => {
    fetchAllPredictions()
    fetchMetrics()
  }, [])

  useEffect(() => {
    if (allData) {
      let filtered = allData.predictions

      if (selectedRegion !== "ALL") {
        filtered = filtered.filter((p) => p.region === selectedRegion)
      }

      if (selectedFlatType !== "ALL") {
        filtered = filtered.filter((p) => p.flat_type === selectedFlatType)
      }

      if (selectedTrend !== "ALL") {
        filtered = filtered.filter((p) => p.trend === selectedTrend)
      }

      setFilteredData(filtered)
    }
  }, [allData, selectedRegion, selectedFlatType, selectedTrend])

  const fetchAllPredictions = async () => {
    setLoading(true)
    setError("")

    try {
      const response = await fetch(`${API_ENDPOINT}/all-predictions`)

      if (!response.ok) {
        throw new Error("Failed to fetch predictions")
      }

      const data: AllPredictionsResponse = await response.json()
      setAllData(data)
      setFilteredData(data.predictions)
    } catch (err) {
      setError("Failed to load predictions. Please try again.")
      console.error("Error:", err)
    } finally {
      setLoading(false)
    }
  }

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_ENDPOINT}/metrics`)

      if (!response.ok) {
        throw new Error("Failed to fetch metrics")
      }

      const data = await response.json()
      setMetrics(data)
    } catch (err) {
      console.error("Error fetching metrics:", err)
    }
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("en-SG", {
      style: "currency",
      currency: "SGD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return "N/A"
    try {
      return new Date(dateString).toLocaleString("en-SG")
    } catch {
      return dateString
    }
  }

  const getPriceChange = (current: number, predicted: number) => {
    const change = ((predicted - current) / current) * 100
    return change
  }

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case "increasing":
        return "text-emerald-400"
      case "decreasing":
        return "text-rose-400"
      case "stable":
        return "text-cyan-400"
      default:
        return "text-slate-400"
    }
  }

  const getTrendBgColor = (trend: string) => {
    switch (trend) {
      case "increasing":
        return "bg-emerald-500/10 border-emerald-500/20"
      case "decreasing":
        return "bg-rose-500/10 border-rose-500/20"
      case "stable":
        return "bg-cyan-500/10 border-cyan-500/20"
      default:
        return "bg-white/5 border-white/10"
    }
  }

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6 mb-12">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-3">
              <h1 className="text-4xl lg:text-5xl font-bold text-white">Price Forecasting</h1>
            </div>
            <p className="text-slate-400 leading-relaxed">
              AI-powered short-term predictions (1-3 months) for HDB resale prices across Singapore
            </p>
          </div>

          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className="backdrop-blur-xl bg-white/5 border border-white/10 hover:bg-white/10 rounded-xl px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 flex items-center gap-2 hover:scale-105 active:scale-95"
          >
            <Activity className="w-4 h-4" />
            {showMetrics ? "Hide" : "View"} Model Performance
          </button>
        </div>

        {showMetrics && metrics && (
          <div className="backdrop-blur-xl bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 rounded-2xl p-6 lg:p-8 mb-8 animate-in fade-in slide-in-from-top duration-300">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                Model Performance Metrics
              </h3>
              <button
                onClick={() => setShowMetrics(false)}
                className="text-slate-400 hover:text-white transition-colors p-1 hover:bg-white/10 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Model Type</p>
                <p className="text-white font-bold text-xl">{metrics.model_type}</p>
              </div>
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Total Models</p>
                <p className="text-white font-bold text-xl">{metrics.num_models}</p>
              </div>
              {metrics.test_performance && (
                <>
                  <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Avg RMSE</p>
                    <p className="text-emerald-400 font-bold text-xl">
                      ${metrics.test_performance.avg_rmse.toLocaleString()}
                    </p>
                  </div>
                  <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-2">Avg MAPE</p>
                    <p className="text-cyan-400 font-bold text-xl">{metrics.test_performance.avg_mape.toFixed(2)}%</p>
                  </div>
                </>
              )}
            </div>
            <div className="mt-4 pt-4 border-t border-white/10">
              <p className="text-slate-500 text-xs">Last Updated: {formatDate(metrics.timestamp)}</p>
            </div>
          </div>
        )}

        <button
          onClick={() => setExpandedPipeline(!expandedPipeline)}
          className="w-full backdrop-blur-xl bg-gradient-to-br from-purple-500/10 via-blue-500/10 to-cyan-500/10 border border-white/10 rounded-2xl p-6 mb-8 hover:scale-[1.01] transition-all duration-300 group"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-left">
                <h2 className="text-xl lg:text-2xl font-bold text-white">Data Pipeline Architecture</h2>
                <p className="text-slate-400 text-sm">Medallion ETL with AWS orchestration</p>
              </div>
            </div>
            <ChevronDown
              className={`w-6 h-6 text-slate-400 transition-transform duration-300 ${expandedPipeline ? "rotate-180" : ""}`}
            />
          </div>
        </button>

        {expandedPipeline && (
          <div className="backdrop-blur-xl bg-gradient-to-br from-purple-500/10 via-blue-500/10 to-cyan-500/10 border border-white/10 rounded-2xl p-6 lg:p-8 mb-8 animate-in fade-in slide-in-from-top duration-300">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="backdrop-blur-xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-xl p-6 hover:scale-105 transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                  <div className="backdrop-blur-xl bg-amber-500/20 border border-amber-500/30 rounded-lg p-3">
                    <Database className="w-6 h-6 text-amber-300" />
                  </div>
                  <span className="text-xs font-bold text-amber-300 backdrop-blur-xl bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
                    LAYER 1
                  </span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Bronze Layer</h3>
                <p className="text-slate-400 text-sm mb-4 leading-relaxed">Raw data ingestion from data.gov.sg API</p>
                <div className="space-y-2">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Fetch HDB resale transactions</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Map towns to regions</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Save to S3 as Parquet</p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-amber-500/20">
                  <p className="text-xs text-slate-500 font-mono">bronze_ingestion.py</p>
                </div>
              </div>

              <div className="backdrop-blur-xl bg-gradient-to-br from-blue-500/10 to-indigo-500/10 border border-blue-500/20 rounded-xl p-6 hover:scale-105 transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                  <div className="backdrop-blur-xl bg-blue-500/20 border border-blue-500/30 rounded-lg p-3">
                    <Layers className="w-6 h-6 text-blue-300" />
                  </div>
                  <span className="text-xs font-bold text-blue-300 backdrop-blur-xl bg-blue-500/10 px-3 py-1 rounded-full border border-blue-500/20">
                    LAYER 2
                  </span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Silver Layer</h3>
                <p className="text-slate-400 text-sm mb-4 leading-relaxed">Feature engineering & aggregation</p>
                <div className="space-y-2">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Aggregate by month/region/flat_type</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Calculate storey midpoints</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Compute remaining lease</p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-blue-500/20">
                  <p className="text-xs text-slate-500 font-mono">silver_features.py</p>
                </div>
              </div>

              <div className="backdrop-blur-xl bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20 rounded-xl p-6 hover:scale-105 transition-all duration-300">
                <div className="flex items-center justify-between mb-4">
                  <div className="backdrop-blur-xl bg-emerald-500/20 border border-emerald-500/30 rounded-lg p-3">
                    <Sparkles className="w-6 h-6 text-emerald-300" />
                  </div>
                  <span className="text-xs font-bold text-emerald-300 backdrop-blur-xl bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20">
                    LAYER 3
                  </span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Gold Layer</h3>
                <p className="text-slate-400 text-sm mb-4 leading-relaxed">ML predictions & serving (1-3 months)</p>
                <div className="space-y-2">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Load SARIMAX models</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Generate 1-3 month forecasts</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5"></div>
                    <p className="text-xs text-slate-300">Serve via DynamoDB & API Gateway</p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-emerald-500/20">
                  <p className="text-xs text-slate-500 font-mono">gold_predictions.py</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
              <div className="backdrop-blur-xl bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 rounded-lg p-4">
                <div className="flex items-center gap-3 mb-3">
                  <div className="backdrop-blur-xl bg-cyan-500/20 border border-cyan-500/30 rounded-lg p-2">
                    <Clock className="w-4 h-4 text-cyan-300" />
                  </div>
                  <h4 className="font-bold text-white">Daily Pipeline</h4>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-300">
                  <span className="backdrop-blur-xl bg-amber-500/10 border border-amber-500/20 px-2 py-1 rounded">
                    Bronze
                  </span>
                  <ArrowRight className="w-3 h-3 text-slate-500" />
                  <span className="backdrop-blur-xl bg-blue-500/10 border border-blue-500/20 px-2 py-1 rounded">
                    Silver
                  </span>
                  <ArrowRight className="w-3 h-3 text-slate-500" />
                  <span className="backdrop-blur-xl bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 rounded">
                    Gold
                  </span>
                </div>
              </div>

              <div className="backdrop-blur-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-lg p-4">
                <div className="flex items-center gap-3 mb-3">
                  <div className="backdrop-blur-xl bg-purple-500/20 border border-purple-500/30 rounded-lg p-2">
                    <Activity className="w-4 h-4 text-purple-300" />
                  </div>
                  <h4 className="font-bold text-white">Weekly Retrain</h4>
                </div>
                <p className="text-xs text-slate-300">Update SARIMAX models incrementally on t4g.nano EC2</p>
              </div>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-4">
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">Infrastructure</p>
                <p className="text-sm font-bold text-white">AWS EC2</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">Storage</p>
                <p className="text-sm font-bold text-white">S3 + DynamoDB</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">ML Framework</p>
                <p className="text-sm font-bold text-white">statsmodels</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">Horizon</p>
                <p className="text-sm font-bold text-white">1-3 months</p>
              </div>
            </div>
          </div>
        )}

        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 lg:p-8 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Region</label>
              <select
                value={selectedRegion}
                onChange={(e) => setSelectedRegion(e.target.value)}
                className="w-full backdrop-blur-xl bg-white/5 border border-white/10 text-white px-4 py-3 rounded-xl focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all hover:bg-white/10"
              >
                <option value="ALL" className="bg-slate-900">
                  All Regions
                </option>
                {allData?.regions.map((region) => (
                  <option key={region} value={region} className="bg-slate-900">
                    {region}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Flat Type</label>
              <select
                value={selectedFlatType}
                onChange={(e) => setSelectedFlatType(e.target.value)}
                className="w-full backdrop-blur-xl bg-white/5 border border-white/10 text-white px-4 py-3 rounded-xl focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all hover:bg-white/10"
              >
                <option value="ALL" className="bg-slate-900">
                  All Types
                </option>
                {allData?.flat_types.map((type) => (
                  <option key={type} value={type} className="bg-slate-900">
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Trend</label>
              <select
                value={selectedTrend}
                onChange={(e) => setSelectedTrend(e.target.value)}
                className="w-full backdrop-blur-xl bg-white/5 border border-white/10 text-white px-4 py-3 rounded-xl focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all hover:bg-white/10"
              >
                <option value="ALL" className="bg-slate-900">
                  All Trends
                </option>
                <option value="increasing" className="bg-slate-900">
                  Increasing
                </option>
                <option value="stable" className="bg-slate-900">
                  Stable
                </option>
                <option value="decreasing" className="bg-slate-900">
                  Decreasing
                </option>
              </select>
            </div>
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-white/10">
            <p className="text-sm text-slate-400">
              Showing <span className="text-white font-semibold">{filteredData.length}</span> of{" "}
              <span className="text-white font-semibold">{allData?.total_count || 0}</span> predictions
            </p>
            {(selectedRegion !== "ALL" || selectedFlatType !== "ALL" || selectedTrend !== "ALL") && (
              <button
                onClick={() => {
                  setSelectedRegion("ALL")
                  setSelectedFlatType("ALL")
                  setSelectedTrend("ALL")
                }}
                className="text-sm text-cyan-400 hover:text-cyan-300 font-medium transition-colors flex items-center gap-1"
              >
                <X className="w-4 h-4" />
                Clear Filters
              </button>
            )}
          </div>
        </div>

        {loading && (
          <div className="flex flex-col items-center justify-center py-24">
            <div className="relative">
              <div className="animate-spin rounded-full h-16 w-16 border-4 border-white/10 border-t-cyan-500"></div>
              <div className="absolute inset-0 rounded-full bg-cyan-500/20 blur-xl"></div>
            </div>
            <p className="text-slate-400 mt-6 font-medium">Loading predictions...</p>
          </div>
        )}

        {error && (
          <div className="backdrop-blur-xl bg-rose-500/10 border border-rose-500/20 rounded-2xl p-6 mb-8">
            <p className="text-rose-300 font-medium">{error}</p>
          </div>
        )}

        {!loading && filteredData.length > 0 && (
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="bg-white/5 border-b border-white/10">
                    <th className="px-6 py-4 text-left text-xs font-bold text-slate-300 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-bold text-slate-300 uppercase tracking-wider">
                      Current
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-bold text-slate-300 uppercase tracking-wider">
                      1 Month
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-bold text-slate-300 uppercase tracking-wider">
                      2 Months
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-bold text-slate-300 uppercase tracking-wider">
                      3 Months
                    </th>
                    <th className="px-6 py-4 text-center text-xs font-bold text-slate-300 uppercase tracking-wider">
                      Trend
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {filteredData.map((pred, idx) => {
                    const change1m = getPriceChange(pred.current_avg_price, pred.predicted_1m_price)
                    const change2m = getPriceChange(pred.current_avg_price, pred.predicted_2m_price)
                    const change3m = getPriceChange(pred.current_avg_price, pred.predicted_3m_price)

                    return (
                      <tr key={idx} className="hover:bg-white/5 transition-colors group">
                        <td className="px-6 py-5 whitespace-nowrap">
                          <div className="text-sm font-bold text-white group-hover:text-cyan-400 transition-colors">
                            {pred.region}
                          </div>
                          <div className="text-xs text-slate-400 font-medium mt-0.5">{pred.flat_type}</div>
                        </td>
                        <td className="px-6 py-5 whitespace-nowrap text-right">
                          <div className="text-base font-bold text-white">{formatPrice(pred.current_avg_price)}</div>
                        </td>
                        <td className="px-6 py-5 whitespace-nowrap text-right">
                          <div className="text-base font-bold text-white">{formatPrice(pred.predicted_1m_price)}</div>
                          <div
                            className={`text-xs font-semibold mt-0.5 ${change1m >= 0 ? "text-emerald-400" : "text-rose-400"}`}
                          >
                            {change1m >= 0 ? "+" : ""}
                            {change1m.toFixed(1)}%
                          </div>
                        </td>
                        <td className="px-6 py-5 whitespace-nowrap text-right">
                          <div className="text-base font-bold text-white">{formatPrice(pred.predicted_2m_price)}</div>
                          <div
                            className={`text-xs font-semibold mt-0.5 ${change2m >= 0 ? "text-emerald-400" : "text-rose-400"}`}
                          >
                            {change2m >= 0 ? "+" : ""}
                            {change2m.toFixed(1)}%
                          </div>
                        </td>
                        <td className="px-6 py-5 whitespace-nowrap text-right">
                          <div className="text-base font-bold text-white">{formatPrice(pred.predicted_3m_price)}</div>
                          <div
                            className={`text-xs font-semibold mt-0.5 ${change3m >= 0 ? "text-emerald-400" : "text-rose-400"}`}
                          >
                            {change3m >= 0 ? "+" : ""}
                            {change3m.toFixed(1)}%
                          </div>
                        </td>
                        <td className="px-6 py-5 whitespace-nowrap text-center">
                          <span
                            className={`inline-flex items-center justify-center px-3 py-1.5 rounded-lg text-xs font-bold backdrop-blur-xl border ${getTrendBgColor(pred.trend)} ${getTrendColor(pred.trend)} capitalize`}
                          >
                            {pred.trend}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!loading && filteredData.length === 0 && allData && (
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-16 text-center">
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
              <Filter className="w-10 h-10 text-slate-400" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">No predictions found</h3>
            <p className="text-slate-400 mb-6">Try adjusting your filters to see more results</p>
            <button
              onClick={() => {
                setSelectedRegion("ALL")
                setSelectedFlatType("ALL")
                setSelectedTrend("ALL")
              }}
              className="backdrop-blur-xl bg-cyan-500/10 border border-cyan-500/20 hover:bg-cyan-500/20 text-cyan-400 px-6 py-3 rounded-xl font-semibold transition-all hover:scale-105 active:scale-95"
            >
              Clear All Filters
            </button>
          </div>
        )}

        <div className="mt-12 pt-8 border-t border-white/10">
          <div className="text-center space-y-2">
            <p className="text-sm text-slate-400">
              Data from <span className="text-cyan-400 font-semibold">data.gov.sg</span> • Last updated:{" "}
              {allData?.last_updated ? formatDate(allData.last_updated) : "N/A"}
            </p>
            <p className="text-xs text-slate-500">
              Powered by SARIMAX Time Series Models • 1-3 Month Predictions • Bronze → Silver → Gold → DynamoDB
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}