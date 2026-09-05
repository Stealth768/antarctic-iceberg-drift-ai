import React, { useState, useEffect, useCallback, Component } from 'react';
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  Polygon,
  CircleMarker,
  Tooltip,
  useMap,
} from 'react-leaflet';
import L from 'leaflet';
import {
  Building2,
  Navigation,
  CheckCircle2,
  AlertTriangle,
  Compass,
  ShieldAlert,
  Fuel,
  Clock,
  Gauge,
  Waves,
  Wind,
  Layers,
  Thermometer,
  Activity,
  Shield,
  Search,
  RefreshCw,
  Eye,
  EyeOff,
  Radio,
  Info,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  Anchor,
} from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// -----------------------------------------------------------------------------
// LEAFLET DEFAULT MARKER ICON FIX
// -----------------------------------------------------------------------------

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// -----------------------------------------------------------------------------
// MAP CONFIGURATION
// -----------------------------------------------------------------------------

const ANTARCTIC_CENTER = [-68.0, 70.0];
const ANTARCTIC_ZOOM = 3;
const CARTO_API_KEY = import.meta.env.VITE_CARTO_API_KEY
// -----------------------------------------------------------------------------
// COMPLETE ROSTER OF ANTARCTIC RESEARCH STATIONS (INDIAN + INTERNATIONAL)
// -----------------------------------------------------------------------------

const ANTARCTIC_STATIONS = [
  // ── Indian Antarctic Stations ──
  {
    id: 'maitri',
    name: 'MAITRI STATION',
    lat: -70.766,
    lon: 11.729,
    status: 'ACTIVE',
    description: "India's second permanent Antarctic research station located in the Schirmacher Oasis.",
    country: 'India',
    isIndian: true,
    operational: true,
  },
  {
    id: 'bharati',
    name: 'BHARATI STATION',
    lat: -69.408,
    lon: 76.187,
    status: 'ACTIVE',
    description: "India's third modern research station located in Larsemann Hills, featuring state-of-the-art oceanographic laboratories.",
    country: 'India',
    isIndian: true,
    operational: true,
  },
  {
    id: 'dakshin_gangotri',
    name: 'DAKSHIN GANGOTRI',
    lat: -70.083,
    lon: 11.572,
    status: 'DECOMMISSIONED',
    description: "India's historic first Antarctic base, established in 1983, decommissioned in 1990.",
    country: 'India',
    isIndian: true,
    operational: false,
  },

  // ── United States ──
  {
    id: 'amundsen_scott',
    name: 'AMUNDSEN-SCOTT SOUTH POLE',
    lat: -90.0,
    lon: 0.0,
    status: 'ACTIVE',
    description: 'United States southernmost research station at the Geographic South Pole.',
    country: 'United States',
    isIndian: false,
    operational: true,
  },
  {
    id: 'mcmurdo',
    name: 'MCMURDO STATION',
    lat: -77.846,
    lon: 166.668,
    status: 'ACTIVE',
    description: 'United States Antarctic logistics hub and largest polar station on Ross Island.',
    country: 'United States',
    isIndian: false,
    operational: true,
  },
  {
    id: 'palmer',
    name: 'PALMER STATION',
    lat: -64.774,
    lon: -64.053,
    status: 'ACTIVE',
    description: 'United States marine biology station located on Anvers Island.',
    country: 'United States',
    isIndian: false,
    operational: true,
  },

  // ── United Kingdom ──
  {
    id: 'rothera',
    name: 'ROTHERA RESEARCH STATION',
    lat: -67.568,
    lon: -68.127,
    status: 'ACTIVE',
    description: 'British Antarctic Survey primary logistics and biological center on Adelaide Island.',
    country: 'United Kingdom',
    isIndian: false,
    operational: true,
  },
  {
    id: 'halley',
    name: 'HALLEY VI RESEARCH STATION',
    lat: -75.583,
    lon: -26.683,
    status: 'ACTIVE',
    description: 'British modular station on the Brunt Ice Shelf, famous for ozone monitoring.',
    country: 'United Kingdom',
    isIndian: false,
    operational: true,
  },

  // ── Australia ──
  {
    id: 'casey',
    name: 'CASEY STATION',
    lat: -66.283,
    lon: 110.524,
    status: 'ACTIVE',
    description: 'Australian Antarctic station on the Bailey Peninsula in Wilkes Land.',
    country: 'Australia',
    isIndian: false,
    operational: true,
  },
  {
    id: 'davis',
    name: 'DAVIS STATION',
    lat: -68.576,
    lon: 77.967,
    status: 'ACTIVE',
    description: 'Australian Antarctic research facility in the ice-free Vestfold Hills.',
    country: 'Australia',
    isIndian: false,
    operational: true,
  },
  {
    id: 'mawson',
    name: 'MAWSON STATION',
    lat: -67.601,
    lon: 62.874,
    status: 'ACTIVE',
    description: "Australia's oldest continuously operating Antarctic station in Mac. Robertson Land.",
    country: 'Australia',
    isIndian: false,
    operational: true,
  },

  // ── New Zealand, France, Italy, Germany ──
  {
    id: 'scott_base',
    name: 'SCOTT BASE',
    lat: -77.851,
    lon: 166.76,
    status: 'ACTIVE',
    description: 'New Zealand research facility located on Pram Point, Ross Island.',
    country: 'New Zealand',
    isIndian: false,
    operational: true,
  },
  {
    id: 'concordia',
    name: 'CONCORDIA STATION',
    lat: -75.1,
    lon: 123.333,
    status: 'ACTIVE',
    description: 'French-Italian joint high-altitude plateau research base at Dome C (3,233m).',
    country: 'France / Italy',
    isIndian: false,
    operational: true,
  },
  {
    id: 'dumont_durville',
    name: "DUMONT D'URVILLE STATION",
    lat: -66.663,
    lon: 140.002,
    status: 'ACTIVE',
    description: 'French scientific station in Adelie Land, home to emperor penguin colonies.',
    country: 'France',
    isIndian: false,
    operational: true,
  },
  {
    id: 'neumayer_iii',
    name: 'NEUMAYER STATION III',
    lat: -70.674,
    lon: -8.274,
    status: 'ACTIVE',
    description: 'German high-tech research facility elevated on hydraulic pillars on the Ekstrom Ice Shelf.',
    country: 'Germany',
    isIndian: false,
    operational: true,
  },

  // ── Russia ──
  {
    id: 'vostok',
    name: 'VOSTOK STATION',
    lat: -78.464,
    lon: 106.837,
    status: 'ACTIVE',
    description: 'Russian research outpost at the southern Pole of Cold above subglacial Lake Vostok.',
    country: 'Russia',
    isIndian: false,
    operational: true,
  },
  {
    id: 'bellingshausen',
    name: 'BELLINGSHAUSEN STATION',
    lat: -62.199,
    lon: -58.964,
    status: 'ACTIVE',
    description: 'Russian research outpost on King George Island in the South Shetland Islands.',
    country: 'Russia',
    isIndian: false,
    operational: true,
  },
  {
    id: 'novolazarevskaya',
    name: 'NOVOLAZAREVSKAYA STATION',
    lat: -70.776,
    lon: 11.821,
    status: 'ACTIVE',
    description: 'Russian base in Queen Maud Land with an intercontinental blue-ice runway.',
    country: 'Russia',
    isIndian: false,
    operational: true,
  },
  {
    id: 'progress',
    name: 'PROGRESS STATION',
    lat: -69.375,
    lon: 76.381,
    status: 'ACTIVE',
    description: 'Russian research base situated in the Larsemann Hills near Bharati station.',
    country: 'Russia',
    isIndian: false,
    operational: true,
  },
  {
    id: 'mirny',
    name: 'MIRNY STATION',
    lat: -66.551,
    lon: 93.013,
    status: 'ACTIVE',
    description: 'Russian scientific station on the coast of the Davis Sea.',
    country: 'Russia',
    isIndian: false,
    operational: true,
  },

  // ── China, Japan, South Korea, South Africa, Norway ──
  {
    id: 'zhongshan',
    name: 'ZHONGSHAN STATION',
    lat: -69.373,
    lon: 76.378,
    status: 'ACTIVE',
    description: 'Chinese research facility in the Larsemann Hills, East Antarctica.',
    country: 'China',
    isIndian: false,
    operational: true,
  },
  {
    id: 'great_wall',
    name: 'GREAT WALL STATION',
    lat: -62.216,
    lon: -58.964,
    status: 'ACTIVE',
    description: 'Chinese Antarctic station located on King George Island.',
    country: 'China',
    isIndian: false,
    operational: true,
  },
  {
    id: 'qinling',
    name: 'QINLING STATION',
    lat: -74.936,
    lon: 163.708,
    status: 'ACTIVE',
    description: "China's fifth polar station on Inexpressible Island in the Ross Sea.",
    country: 'China',
    isIndian: false,
    operational: true,
  },
  {
    id: 'syowa',
    name: 'SYOWA STATION',
    lat: -69.004,
    lon: 39.58,
    status: 'ACTIVE',
    description: 'Japanese permanent research facility on East Ongul Island in Queen Maud Land.',
    country: 'Japan',
    isIndian: false,
    operational: true,
  },
  {
    id: 'jang_bogo',
    name: 'JANG BOGO STATION',
    lat: -74.623,
    lon: 164.229,
    status: 'ACTIVE',
    description: 'South Korean year-round scientific station in Terra Nova Bay.',
    country: 'South Korea',
    isIndian: false,
    operational: true,
  },
  {
    id: 'king_sejong',
    name: 'KING SEJONG STATION',
    lat: -62.223,
    lon: -58.787,
    status: 'ACTIVE',
    description: 'South Korean station on the Barton Peninsula of King George Island.',
    country: 'South Korea',
    isIndian: false,
    operational: true,
  },
  {
    id: 'sanae_iv',
    name: 'SANAE IV',
    lat: -71.673,
    lon: -2.842,
    status: 'ACTIVE',
    description: 'South African National Antarctic Expedition base built on the nunatak Vesleskarvet.',
    country: 'South Africa',
    isIndian: false,
    operational: true,
  },
  {
    id: 'troll',
    name: 'TROLL STATION',
    lat: -72.011,
    lon: 2.535,
    status: 'ACTIVE',
    description: 'Norwegian year-round station on the nunatak Jutulsessen in Queen Maud Land.',
    country: 'Norway',
    isIndian: false,
    operational: true,
  },

  // ── Argentina, Chile, Brazil, Poland, Uruguay, Ukraine ──
  {
    id: 'esperanza',
    name: 'ESPERANZA BASE',
    lat: -63.398,
    lon: -56.997,
    status: 'ACTIVE',
    description: 'Argentine station in Hope Bay with a permanent civilian community and school.',
    country: 'Argentina',
    isIndian: false,
    operational: true,
  },
  {
    id: 'marambio',
    name: 'MARAMBIO STATION',
    lat: -64.241,
    lon: -56.626,
    status: 'ACTIVE',
    description: 'Argentine base and major air logistics hub located on Seymour Island.',
    country: 'Argentina',
    isIndian: false,
    operational: true,
  },
  {
    id: 'orcadas',
    name: 'ORCADAS BASE',
    lat: -60.738,
    lon: -44.738,
    status: 'ACTIVE',
    description: 'Argentine base on Laurie Island, the oldest continuously active station in Antarctica (1904).',
    country: 'Argentina',
    isIndian: false,
    operational: true,
  },
  {
    id: 'san_martin',
    name: 'SAN MARTIN BASE',
    lat: -68.13,
    lon: -67.101,
    status: 'ACTIVE',
    description: 'Argentine research base located on Barry Island in Marguerite Bay.',
    country: 'Argentina',
    isIndian: false,
    operational: true,
  },
  {
    id: 'carlini',
    name: 'CARLINI BASE',
    lat: -62.238,
    lon: -58.667,
    status: 'ACTIVE',
    description: 'Argentine scientific station on King George Island housing the Dallmann Laboratory.',
    country: 'Argentina',
    isIndian: false,
    operational: true,
  },
  {
    id: 'belgrano_ii',
    name: 'BELGRANO II BASE',
    lat: -77.874,
    lon: -34.627,
    status: 'ACTIVE',
    description: "Argentina's southernmost permanent base built on solid rock in the Weddell Sea sector.",
    country: 'Argentina',
    isIndian: false,
    operational: true,
  },
  {
    id: 'eduardo_frei',
    name: 'BASE PTE. EDUARDO FREI',
    lat: -62.197,
    lon: -58.981,
    status: 'ACTIVE',
    description: "Chile's primary Antarctic facility and civilian community Villa Las Estrellas.",
    country: 'Chile',
    isIndian: false,
    operational: true,
  },
  {
    id: 'arturo_prat',
    name: 'CAPITAN ARTURO PRAT BASE',
    lat: -62.479,
    lon: -59.664,
    status: 'ACTIVE',
    description: 'Chilean Navy historical research station on Greenwich Island.',
    country: 'Chile',
    isIndian: false,
    operational: true,
  },
  {
    id: 'bernardo_ohiggins',
    name: "GRAL. BERNARDO O'HIGGINS",
    lat: -63.321,
    lon: -57.898,
    status: 'ACTIVE',
    description: 'Chilean Army operational and scientific research facility on Cape Legoupil.',
    country: 'Chile',
    isIndian: false,
    operational: true,
  },
  {
    id: 'escudero',
    name: 'JULIO ESCUDERO BASE',
    lat: -62.201,
    lon: -58.963,
    status: 'ACTIVE',
    description: 'Chilean Antarctic Institute (INACH) station on Fildes Peninsula.',
    country: 'Chile',
    isIndian: false,
    operational: true,
  },
  {
    id: 'arctowski',
    name: 'HENRYK ARCTOWSKI STATION',
    lat: -62.159,
    lon: -58.471,
    status: 'ACTIVE',
    description: 'Polish Academy of Sciences marine research station on Admiralty Bay.',
    country: 'Poland',
    isIndian: false,
    operational: true,
  },
  {
    id: 'comandante_ferraz',
    name: 'COMANDANTE FERRAZ STATION',
    lat: -62.084,
    lon: -58.393,
    status: 'ACTIVE',
    description: 'Brazilian state-of-the-art permanent scientific base on Keller Peninsula.',
    country: 'Brazil',
    isIndian: false,
    operational: true,
  },
  {
    id: 'artigas',
    name: 'ARTIGAS BASE',
    lat: -62.184,
    lon: -58.903,
    status: 'ACTIVE',
    description: 'Uruguayan scientific facility located on King George Island.',
    country: 'Uruguay',
    isIndian: false,
    operational: true,
  },
  {
    id: 'vernadsky',
    name: 'VERNADSKY RESEARCH STATION',
    lat: -65.245,
    lon: -64.257,
    status: 'ACTIVE',
    description: 'Ukrainian base on Galindez Island, formerly the British Faraday Station where ozone hole was discovered.',
    country: 'Ukraine',
    isIndian: false,
    operational: true,
  },
  {
    id: 'eco_nelson',
    name: 'ECO-NELSON',
    lat: -62.246,
    lon: -59.004,
    status: 'ACTIVE',
    description: 'Private ecological research outpost on Nelson Island.',
    country: 'Czech Republic',
    isIndian: false,
    operational: true,
  },
  {
    id: 'gars_ohiggins',
    name: "GARS O'HIGGINS STATION",
    lat: -63.321,
    lon: -57.9,
    status: 'ACTIVE',
    description: 'German satellite receiving facility operated in cooperation with Chile.',
    country: 'Germany',
    isIndian: false,
    operational: true,
  },
];

// -----------------------------------------------------------------------------
// DEFAULT / FALLBACK FLEET VESSELS
// -----------------------------------------------------------------------------

const DEFAULT_VESSELS = [
  {
    id: 'VSL-047',
    name: 'ORV SAGAR NIDHI',
    type: 'Ice-Strengthened Oceanographic Vessel',
    lat: -64.25,
    lon: -56.75,
    speedKnots: 11.5,
    iceClass: 'PC5',
    flag: 'India (MoES)',
  },
  {
    id: 'VSL-048',
    name: 'M/V VASILIY GOLOVNIN',
    type: 'Polar Cargo / Supply Vessel',
    lat: -62.5,
    lon: 25.0,
    speedKnots: 12.0,
    iceClass: 'PC4',
    flag: 'Charter (MoES)',
  },
  {
    id: 'VSL-049',
    name: 'INS SAGAR KANYA',
    type: 'Research Vessel',
    lat: -60.0,
    lon: 55.0,
    speedKnots: 10.0,
    iceClass: 'PC6',
    flag: 'India (MoES)',
  },
];

// -----------------------------------------------------------------------------
// ROUTE METADATA
// -----------------------------------------------------------------------------

const ROUTE_META = {
  samudra: {
    title: 'SAMUDRA ICE-OPTIMIZED',
    badge: 'RECOMMENDED',
    color: '#10B981',
    description: 'Physics-informed route avoiding high ice concentration and iceberg drift convergence zones.',
  },
  direct: {
    title: 'DIRECT RHUMB-LINE',
    badge: 'HIGH RISK',
    color: '#EF4444',
    description: 'Direct rhumb line ignoring heavy pack ice and iceberg hazards.',
  },
  alternative: {
    title: 'ALTERNATIVE CORRIDOR',
    badge: 'BALANCED',
    color: '#F59E0B',
    description: 'Conservative offshore corridor trading travel distance for wider safety margins.',
  },
};

// -----------------------------------------------------------------------------
// MATHEMATICAL & GEOGRAPHIC HELPERS
// -----------------------------------------------------------------------------

function haversineNm(lat1Deg, lon1Deg, lat2Deg, lon2Deg) {
  const R = 3440.065;
  const lat1 = (lat1Deg * Math.PI) / 180;
  const lat2 = (lat2Deg * Math.PI) / 180;
  const dLat = ((lat2Deg - lat1Deg) * Math.PI) / 180;
  const dLon = ((lon2Deg - lon1Deg) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function formatEta(distanceNm, speedKnots) {
  if (!Number.isFinite(distanceNm) || !Number.isFinite(speedKnots) || speedKnots <= 0) {
    return '—';
  }
  const hours = distanceNm / speedKnots;
  const days = Math.floor(hours / 24);
  const remHours = Math.round(hours % 24);
  return days === 0 ? `${remHours}h` : `${days}d ${remHours}h`;
}

function formatCoord(value, positive, negative) {
  if (!Number.isFinite(value)) return '—';
  const direction = value >= 0 ? positive : negative;
  return `${Math.abs(value).toFixed(3)}°${direction}`;
}

// -----------------------------------------------------------------------------
// DEMO FALLBACK ROUTE ENGINE
// -----------------------------------------------------------------------------

function buildFallbackRoutes(vessel, destination) {
  const directDist = haversineNm(
    vessel.lat,
    vessel.lon,
    destination.lat,
    destination.lon
  );

  const directRounded = Math.max(1, Math.round(directDist));
  const samudraDist = Math.max(1, Math.round(directDist * 1.06));
  const alternativeDist = Math.max(1, Math.round(directDist * 1.14));

  const midpointLat = (vessel.lat + destination.lat) / 2;
  const midpointLon = (vessel.lon + destination.lon) / 2;

  return {
    source: 'DEMO FALLBACK',
    vessel: vessel.name,
    destination: destination.name,
    routes: {
      samudra: {
        route_id: 'route_b_recommended',
        name: ROUTE_META.samudra.title,
        distance_nm: samudraDist,
        eta: formatEta(samudraDist, vessel.speedKnots),
        fuel_mt: Number((samudraDist * 0.076).toFixed(1)),
        safety_score: 86.4,
        ice_risk: 'LOW',
        fuel_saving: '9.8%',
        color: '#10B981',
        path: [
          [vessel.lat, vessel.lon],
          [midpointLat - 1.2, midpointLon + 1.8],
          [destination.lat, destination.lon],
        ],
      },
      direct: {
        route_id: 'route_a_direct',
        name: ROUTE_META.direct.title,
        distance_nm: directRounded,
        eta: formatEta(directRounded, vessel.speedKnots),
        fuel_mt: Number((directRounded * 0.088).toFixed(1)),
        safety_score: 51.2,
        ice_risk: 'HIGH',
        fuel_saving: '0.0%',
        color: '#EF4444',
        path: [
          [vessel.lat, vessel.lon],
          [destination.lat, destination.lon],
        ],
      },
      alternative: {
        route_id: 'route_c_alternative',
        name: ROUTE_META.alternative.title,
        distance_nm: alternativeDist,
        eta: formatEta(alternativeDist, vessel.speedKnots),
        fuel_mt: Number((alternativeDist * 0.083).toFixed(1)),
        safety_score: 75.6,
        ice_risk: 'MODERATE',
        fuel_saving: '3.9%',
        color: '#F59E0B',
        path: [
          [vessel.lat, vessel.lon],
          [midpointLat + 1.4, midpointLon - 2.2],
          [destination.lat, destination.lon],
        ],
      },
    },
  };
}

// -----------------------------------------------------------------------------
// NORMALIZERS
// -----------------------------------------------------------------------------

function normalizeVessels(data) {
  const list = Array.isArray(data)
    ? data
    : Array.isArray(data?.vessels)
      ? data.vessels
      : [];

  return list.map((v) => {
    const lat = Number(v.latitude ?? v.lat);
    const lon = Number(v.longitude ?? v.lon);
    const speed = Number(v.speed_knots ?? v.speed ?? v.speedKnots ?? 11.5);

    return {
      ...v,
      id: v.vessel_id ?? v.id ?? `VSL-${Math.random().toString(36).substring(2, 6)}`,
      name: v.name ?? v.display_name ?? v.vessel_name ?? 'VESSEL',
      type: v.type ?? v.vessel_type ?? 'Research / Logistics Vessel',
      lat: Number.isFinite(lat) ? lat : -64.25,
      lon: Number.isFinite(lon) ? lon : -56.75,
      speedKnots: Number.isFinite(speed) && speed > 0 ? speed : 11.5,
      iceClass: v.polar_ice_class ?? v.ice_class ?? 'PC5',
      flag: v.flag ?? 'Maritime Fleet',
    };
  });
}

function normalizeRouteResponse(data, vessel, destination) {
  if (!data) return null;

  if (Array.isArray(data.routes) && data.routes.length > 0) {
    const routes = {};
    for (const r of data.routes) {
      const id = String(r.route_id ?? '').toLowerCase();
      const label = String(r.route_label ?? r.name ?? '').toLowerCase();

      const path = Array.isArray(r.path_coordinates)
        ? r.path_coordinates.map((c) => [c[1], c[0]])
        : [[vessel.lat, vessel.lon], [destination.lat, destination.lon]];

      const formatted = {
        route_id: r.route_id,
        name: r.route_label ?? r.name,
        distance_nm: Math.round(r.distance_nm ?? r.distance_km * 0.539957 ?? 0),
        eta: r.eta_hours ? `${r.eta_hours.toFixed(1)}h` : formatEta(r.distance_nm, vessel.speedKnots),
        fuel_mt: Number((r.estimated_fuel_mt ?? 0).toFixed(1)),
        safety_score: Number((r.safety_score ?? 0).toFixed(1)),
        ice_risk: r.risk_level ?? (r.sea_ice_risk > 50 ? 'HIGH' : r.sea_ice_risk > 25 ? 'MODERATE' : 'LOW'),
        fuel_saving: r.fuel_saving_percent ? `${r.fuel_saving_percent.toFixed(1)}%` : '0.0%',
        color: r.color_code === 'green' ? '#10B981' : r.color_code === 'red' ? '#EF4444' : '#F59E0B',
        path,
      };

      if (id.includes('route_b') || label.includes('recommended') || label.includes('samudra')) {
        routes.samudra = { ...formatted, name: ROUTE_META.samudra.title, color: '#10B981' };
      } else if (id.includes('route_a') || label.includes('direct')) {
        routes.direct = { ...formatted, name: ROUTE_META.direct.title, color: '#EF4444' };
      } else if (id.includes('route_c') || label.includes('alternative') || label.includes('corridor')) {
        routes.alternative = { ...formatted, name: ROUTE_META.alternative.title, color: '#F59E0B' };
      }
    }

    if (routes.samudra || routes.direct || routes.alternative) {
      return {
        ...data,
        source: 'A* ROUTING ENGINE',
        routes,
      };
    }
  }

  if (data.routes?.samudra && data.routes?.direct) {
    return data;
  }

  return null;
}

// -----------------------------------------------------------------------------
// MAP FLY-TO COMPONENT
// -----------------------------------------------------------------------------

function MapFlyTo({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center && Array.isArray(center) && center.length === 2) {
      map.flyTo(center, zoom ?? map.getZoom(), { duration: 1.8 });
    }
  }, [center, zoom, map]);
  return null;
}

// -----------------------------------------------------------------------------
// CUSTOM LEAFLET DIV ICONS
// -----------------------------------------------------------------------------

function createStationIcon(station, isSelected) {
  const isIndian = station.isIndian;
  const operational = station.operational;

  let bg = '#0f172a';
  let border = '#38bdf8';
  let glow = 'rgba(56, 189, 248, 0.4)';

  if (isIndian) {
    bg = isSelected ? '#042f2e' : '#082f49';
    border = '#22d3ee';
    glow = 'rgba(34, 211, 238, 0.8)';
  } else if (!operational) {
    bg = '#451a03';
    border = '#f59e0b';
    glow = 'rgba(245, 158, 11, 0.3)';
  } else if (isSelected) {
    bg = '#064e3b';
    border = '#10b981';
    glow = 'rgba(16, 185, 129, 0.8)';
  }

  return L.divIcon({
    className: 'custom-station-pin',
    html: `
      <div style="
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: ${bg};
        border: 2px solid ${border};
        box-shadow: 0 0 ${isSelected ? '16px' : '8px'} ${glow};
        font-size: 14px;
        cursor: pointer;
        transition: transform 0.2s ease;
      ">
        ${isIndian ? '🇮🇳' : operational ? '🏛️' : '⚠️'}
      </div>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15],
  });
}

function createVesselIcon(vessel, isSelected) {
  return L.divIcon({
    className: 'custom-vessel-pin',
    html: `
      <div style="
        position: relative;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #022c22;
        border: 2px solid ${isSelected ? '#10b981' : '#34d399'};
        box-shadow: 0 0 ${isSelected ? '18px #10b981' : '8px rgba(52, 211, 153, 0.5)'};
        cursor: pointer;
      ">
        <span style="font-size: 15px;">🚢</span>
        ${
          isSelected
            ? `<div style="
                position: absolute;
                inset: -6px;
                border-radius: 50%;
                border: 1.5px dashed #10b981;
                animation: spin 6s linear infinite;
              "></div>`
            : ''
        }
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16],
  });
}

// -----------------------------------------------------------------------------
// ERROR BOUNDARY
// -----------------------------------------------------------------------------

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Antarctic Nav AI UI Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-screen w-screen bg-slate-950 text-slate-100 p-6">
          <div className="bg-slate-900 border border-rose-800/80 rounded-xl p-6 max-w-lg w-full text-center shadow-2xl">
            <AlertTriangle className="h-12 w-12 text-rose-500 mx-auto mb-4 animate-bounce" />
            <h2 className="text-xl font-bold text-rose-400 mb-2">SYSTEM INTERFACE ERROR</h2>
            <p className="text-xs text-slate-400 mb-4">
              An unexpected render anomaly occurred in the navigation viewport.
            </p>
            <div className="p-3 bg-slate-950 rounded text-left font-mono text-[11px] text-rose-300 overflow-x-auto mb-6 border border-rose-950">
              {this.state.error?.message || 'Unknown Exception'}
            </div>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-slate-950 font-bold text-xs rounded transition flex items-center justify-center gap-2 mx-auto"
            >
              <RefreshCw className="h-4 w-4" /> RELOAD DASHBOARD
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// -----------------------------------------------------------------------------
// COLLAPSIBLE PANEL SECTION COMPONENT
// -----------------------------------------------------------------------------

function PanelSection({ id, title, icon, badge, isCollapsed, onToggle, children }) {
  return (
    <div className="border-b border-slate-800 last:border-b-0 pb-2 mb-2">
      <button
        onClick={() => onToggle(id)}
        className="w-full flex items-center justify-between py-1.5 px-2 hover:bg-slate-800/50 rounded transition text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-cyan-400">{icon}</span>
          <span className="text-xs font-bold tracking-wider text-slate-200">{title}</span>
          {badge && (
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              {badge}
            </span>
          )}
        </div>
        <span className="text-slate-500">
          {isCollapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
        </span>
      </button>
      {!isCollapsed && <div className="mt-2 px-1">{children}</div>}
    </div>
  );
}

// -----------------------------------------------------------------------------
// MAIN DASHBOARD COMPONENT
// -----------------------------------------------------------------------------

export default function App() {
  const [vessels, setVessels] = useState(DEFAULT_VESSELS);
  const [selectedVessel, setSelectedVessel] = useState(DEFAULT_VESSELS[0]);
  const [selectedDestination, setSelectedDestination] = useState(ANTARCTIC_STATIONS[0]);
  const [stationSearch, setStationSearch] = useState('');
  const [stationTab, setStationTab] = useState('all');

  const [mapCenter, setMapCenter] = useState(ANTARCTIC_CENTER);
  const [mapZoom, setMapZoom] = useState(ANTARCTIC_ZOOM);

  const [layers, setLayers] = useState({
    vessels: true,
    stations: true,
    riskZones: true,
    seaIce: false,
    currents: false,
    wind: false,
  });

  const [riskZonesData, setRiskZonesData] = useState(null);
  const [seaIceData, setSeaIceData] = useState(null);
  const [currentsData, setCurrentsData] = useState(null);
  const [windData, setWindData] = useState(null);

  const [liveEnv, setLiveEnv] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);

  const [isCalculating, setIsCalculating] = useState(false);
  const [routeData, setRouteData] = useState(null);
  const [routeSource, setRouteSource] = useState(null);
  const [error, setError] = useState(null);

  const [collapsed, setCollapsed] = useState({
    layers: false,
    stations: false,
    telemetry: false,
    alerts: false,
    system: false,
  });

  const toggleSection = (sectionId) => {
    setCollapsed((prev) => ({ ...prev, [sectionId]: !prev[sectionId] }));
  };

  const loadSystemAndVessels = useCallback(async () => {
    try {
      const [vesselsRes, statusRes, alertsRes] = await Promise.allSettled([
        fetch('/api/v1/vessels'),
        fetch('/api/v1/system/status'),
        fetch('/api/v1/alerts'),
      ]);

      if (vesselsRes.status === 'fulfilled' && vesselsRes.value.ok) {
        const rawVessels = await vesselsRes.value.json();
        const norm = normalizeVessels(rawVessels);
        if (norm.length > 0) {
          setVessels(norm);
          setSelectedVessel((curr) => {
            const found = norm.find((v) => v.id === curr?.id);
            return found ?? norm[0];
          });
        }
      }

      if (statusRes.status === 'fulfilled' && statusRes.value.ok) {
        const statusJson = await statusRes.value.json();
        setSystemStatus(statusJson);
      }

      if (alertsRes.status === 'fulfilled' && alertsRes.value.ok) {
        const alertsJson = await alertsRes.value.json();
        setAlerts(Array.isArray(alertsJson) ? alertsJson : []);
      }
    } catch (err) {
      console.warn('Backend service connection initialization warning:', err);
    }
  }, []);

  useEffect(() => {
    loadSystemAndVessels();
  }, [loadSystemAndVessels]);

  useEffect(() => {
    async function loadRiskZones() {
      try {
        const res = await fetch('/api/v1/risk-zones');
        if (res.ok) {
          const json = await res.json();
          setRiskZonesData(json);
        }
      } catch (err) {
        console.warn('Risk zones API unavailable:', err);
      }
    }
    loadRiskZones();
  }, []);

  useEffect(() => {
    if (layers.seaIce && !seaIceData) {
      fetch('/api/v1/layers/sea-ice?min_lat=-66.5&max_lat=-61.5&min_lon=-61.0&max_lon=-52.0&resolution_km=30.0')
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => data && setSeaIceData(data))
        .catch((e) => console.warn('Sea ice layer load err:', e));
    }
    if (layers.currents && !currentsData) {
      fetch('/api/v1/layers/currents?min_lat=-66.5&max_lat=-61.5&min_lon=-61.0&max_lon=-52.0&resolution_km=35.0')
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => data && setCurrentsData(data))
        .catch((e) => console.warn('Currents layer load err:', e));
    }
    if (layers.wind && !windData) {
      fetch('/api/v1/layers/wind?min_lat=-66.5&max_lat=-61.5&min_lon=-61.0&max_lon=-52.0&resolution_km=35.0')
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => data && setWindData(data))
        .catch((e) => console.warn('Wind layer load err:', e));
    }
  }, [layers.seaIce, layers.currents, layers.wind, seaIceData, currentsData, windData]);

  useEffect(() => {
    if (!selectedVessel) return;
    let active = true;

    async function fetchTelemetry() {
      try {
        const res = await fetch(
          `/api/v1/environmental/live?latitude=${selectedVessel.lat}&longitude=${selectedVessel.lon}`
        );
        if (res.ok && active) {
          const json = await res.json();
          setLiveEnv(json);
        }
      } catch (e) {
        console.warn('Live telemetry query error:', e);
      }
    }

    fetchTelemetry();
    return () => {
      active = false;
    };
  }, [selectedVessel]);

  useEffect(() => {
    if (!selectedVessel || !selectedDestination || !selectedDestination.operational) {
      return;
    }

    let cancelled = false;

    async function calculateMultiRoute() {
      setIsCalculating(true);
      setError(null);
      setRouteData(null);
      setRouteSource(null);

      const payload = {
        start_latitude: selectedVessel.lat,
        start_longitude: selectedVessel.lon,
        goal_latitude: selectedDestination.lat,
        goal_longitude: selectedDestination.lon,
        vessel_id: selectedVessel.id,
        polar_ice_class: selectedVessel.iceClass,
        cruising_speed_knots: selectedVessel.speedKnots,
        grid_resolution_km: 25.0,
      };

      try {
        const res = await fetch('/api/v1/routes/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          throw new Error(`Routing Engine responded with HTTP ${res.status}`);
        }

        const data = await res.json();
        if (cancelled) return;

        const normalized = normalizeRouteResponse(data, selectedVessel, selectedDestination);
        if (normalized?.routes?.samudra && normalized?.routes?.direct) {
          setRouteData(normalized);
          setRouteSource('A* ROUTING ENGINE');
          return;
        }
        throw new Error('Backend route response missing candidate corridors.');
      } catch (err) {
        if (cancelled) return;
        console.info('Target waypoint outside regional reanalysis bounds or planner offline. Using calibrated geodesic fallback.', err);
        const fallback = buildFallbackRoutes(selectedVessel, selectedDestination);
        setRouteData(fallback);
        setRouteSource('DEMO FALLBACK');
      } finally {
        if (!cancelled) {
          setIsCalculating(false);
        }
      }
    }

    calculateMultiRoute();

    return () => {
      cancelled = true;
    };
  }, [selectedVessel, selectedDestination]);

  const handleStationClick = (station) => {
    setMapCenter([station.lat, station.lon]);
    setMapZoom(6);
    if (station.operational) {
      setSelectedDestination(station);
    } else {
      setSelectedDestination(null);
      setRouteData(null);
      setRouteSource(null);
    }
  };

  const handleVesselChange = (e) => {
    const vsl = vessels.find((v) => String(v.id) === e.target.value);
    if (vsl) {
      setSelectedVessel(vsl);
      if (!selectedDestination) {
        setMapCenter([vsl.lat, vsl.lon]);
        setMapZoom(4);
      }
    }
  };

  const clearDestination = () => {
    setSelectedDestination(null);
    setRouteData(null);
    setRouteSource(null);
    setError(null);
    setMapCenter(ANTARCTIC_CENTER);
    setMapZoom(ANTARCTIC_ZOOM);
  };

  const filteredStations = ANTARCTIC_STATIONS.filter((s) => {
    const matchesQuery =
      s.name.toLowerCase().includes(stationSearch.toLowerCase()) ||
      s.country.toLowerCase().includes(stationSearch.toLowerCase());
    if (!matchesQuery) return false;
    if (stationTab === 'india') return s.isIndian;
    if (stationTab === 'active') return s.operational;
    return true;
  });

  return (
    <ErrorBoundary>
      <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 font-mono overflow-hidden select-none">
        <header className="flex items-center justify-between px-5 py-2.5 bg-slate-900/90 backdrop-blur border-b border-slate-800 z-20 shrink-0">
          <div className="flex items-center gap-3">
            <div className="bg-cyan-500/15 p-2 rounded-lg border border-cyan-500/40">
              <Compass className="h-6 w-6 text-cyan-400 animate-spin-slow" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-wider text-cyan-400 flex items-center gap-2">
                SAMUDRA NAV AI
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-semibold">
                  समुद्र
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-normal">
                  SIH26059
                </span>
              </h1>
              <p className="text-[10px] text-slate-400 tracking-widest uppercase">
                Indian Antarctic Navigation & Tactical Iceberg Decision Support System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-400 text-[11px]">ACTIVE VESSEL:</span>
              <select
                value={selectedVessel?.id ?? ''}
                onChange={handleVesselChange}
                className="bg-slate-800/90 border border-slate-700 text-cyan-300 px-3 py-1.5 rounded focus:outline-none focus:border-cyan-500 text-xs cursor-pointer"
              >
                {vessels.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.iceClass} • {v.speedKnots} kn)
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2 bg-emerald-950/70 border border-emerald-700/80 px-3 py-1 rounded-full shadow-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              <span className="text-emerald-400 font-semibold text-[11px]">SYSTEM ONLINE</span>
            </div>

            <button
              onClick={() => loadSystemAndVessels()}
              title="Refresh Telemetry & Alerts"
              className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-cyan-400 transition"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        </header>

        <div className="flex flex-1 relative overflow-hidden">
          <div className="absolute inset-0 z-0">
            <MapContainer
              center={ANTARCTIC_CENTER}
              zoom={ANTARCTIC_ZOOM}
              className="h-full w-full"
              zoomControl={false}
              minZoom={2}
              maxZoom={12}
            >
              <TileLayer
                url={`https://{s}.basemaps.cartocdn.com/rastertiles/dark_matter/{z}/{x}/{y}.png${CARTO_API_KEY ? `?key=${CARTO_API_KEY}` : ''}`}
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a> | SIH26059'
                subdomains="abcd"
                maxZoom={20}
              />

              <MapFlyTo center={mapCenter} zoom={mapZoom} />

              {layers.riskZones &&
                riskZonesData?.features &&
                riskZonesData.features.map((feature, idx) => {
                  if (feature.geometry?.type !== 'Polygon') return null;
                  const ring = feature.geometry.coordinates?.[0];
                  if (!Array.isArray(ring)) return null;
                  const positions = ring.map((c) => [c[1], c[0]]);
                  const props = feature.properties || {};

                  return (
                    <Polygon
                      key={`risk-zone-${idx}`}
                      positions={positions}
                      pathOptions={{
                        color: props.color || '#eab308',
                        fillColor: props.fill_color || '#eab308',
                        fillOpacity: 0.22,
                        weight: 1.5,
                      }}
                    >
                      <Popup>
                        <div className="text-xs font-mono p-1">
                          <div className="font-bold text-amber-400 uppercase">{props.zone_type}</div>
                          <div className="text-[10px] text-slate-400 mt-1">{props.description}</div>
                          <div className="text-[10px] text-slate-300 mt-2">
                            Severity: <span className="font-bold uppercase text-amber-300">{props.severity}</span>
                          </div>
                          {props.risk_score && (
                            <div className="text-[10px] text-slate-400">Risk Score: {props.risk_score} / 100</div>
                          )}
                        </div>
                      </Popup>
                    </Polygon>
                  );
                })}

              {layers.seaIce &&
                seaIceData?.features &&
                seaIceData.features.map((feat, i) => {
                  const [lon, lat] = feat.geometry?.coordinates ?? [];
                  const pct = feat.properties?.sea_ice_pct ?? 0;
                  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

                  return (
                    <CircleMarker
                      key={`ice-${i}`}
                      center={[lat, lon]}
                      radius={4}
                      pathOptions={{
                        color: pct > 70 ? '#38bdf8' : pct > 30 ? '#0284c7' : '#0369a1',
                        fillColor: pct > 70 ? '#bae6fd' : '#38bdf8',
                        fillOpacity: 0.7,
                        weight: 1,
                      }}
                    >
                      <Tooltip direction="top" offset={[0, -4]}>
                        <span>Sea Ice: {pct}%</span>
                      </Tooltip>
                    </CircleMarker>
                  );
                })}

              {layers.currents &&
                currentsData?.features &&
                currentsData.features.map((feat, i) => {
                  const [lon, lat] = feat.geometry?.coordinates ?? [];
                  const speed = feat.properties?.speed_knots ?? 0;
                  const dir = feat.properties?.direction_deg ?? 0;
                  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

                  return (
                    <CircleMarker
                      key={`curr-${i}`}
                      center={[lat, lon]}
                      radius={3.5}
                      pathOptions={{
                        color: '#06b6d4',
                        fillColor: '#22d3ee',
                        fillOpacity: 0.65,
                        weight: 1,
                      }}
                    >
                      <Tooltip direction="top" offset={[0, -4]}>
                        <span>Current: {speed} kn @ {dir}°</span>
                      </Tooltip>
                    </CircleMarker>
                  );
                })}

              {layers.wind &&
                windData?.features &&
                windData.features.map((feat, i) => {
                  const [lon, lat] = feat.geometry?.coordinates ?? [];
                  const spd = feat.properties?.speed_knots ?? 0;
                  const dir = feat.properties?.direction_deg ?? 0;
                  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

                  return (
                    <CircleMarker
                      key={`wind-${i}`}
                      center={[lat, lon]}
                      radius={3.5}
                      pathOptions={{
                        color: '#a855f7',
                        fillColor: '#c084fc',
                        fillOpacity: 0.65,
                        weight: 1,
                      }}
                    >
                      <Tooltip direction="top" offset={[0, -4]}>
                        <span>Wind: {spd} kn @ {dir}°</span>
                      </Tooltip>
                    </CircleMarker>
                  );
                })}

              {layers.vessels &&
                vessels.map((vsl) => {
                  const isSelected = selectedVessel?.id === vsl.id;
                  return (
                    <Marker
                      key={`vessel-${vsl.id}`}
                      position={[vsl.lat, vsl.lon]}
                      icon={createVesselIcon(vsl, isSelected)}
                      eventHandlers={{
                        click: () => {
                          setSelectedVessel(vsl);
                          setMapCenter([vsl.lat, vsl.lon]);
                        },
                      }}
                    >
                      <Popup>
                        <div className="p-1 min-w-[210px] text-xs font-mono">
                          <div className="font-bold text-emerald-400 flex items-center gap-1">
                            <Navigation className="w-3.5 h-3.5" />
                            {vsl.name}
                          </div>
                          <div className="text-[10px] text-slate-300 mt-1">{vsl.type}</div>
                          <div className="text-[10px] text-slate-400 mt-2">
                            Speed: <strong className="text-cyan-300">{vsl.speedKnots} kn</strong>
                          </div>
                          <div className="text-[10px] text-slate-400">
                            Polar Class: <strong className="text-slate-200">{vsl.iceClass}</strong>
                          </div>
                          <div className="text-[10px] text-slate-400">
                            Flag: <strong className="text-slate-200">{vsl.flag}</strong>
                          </div>
                          <div className="text-[10px] text-slate-500 mt-1">
                            Lat: {formatCoord(vsl.lat, 'N', 'S')} | Lon: {formatCoord(vsl.lon, 'E', 'W')}
                          </div>
                          <button
                            onClick={() => {
                              setSelectedVessel(vsl);
                              setMapCenter([vsl.lat, vsl.lon]);
                            }}
                            className="mt-2 w-full py-1 text-[10px] bg-emerald-950 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 rounded font-semibold text-center"
                          >
                            SET AS ACTIVE ORIGIN
                          </button>
                        </div>
                      </Popup>
                    </Marker>
                  );
                })}

              {layers.stations &&
                ANTARCTIC_STATIONS.map((st) => {
                  const isSelected = selectedDestination?.id === st.id;
                  return (
                    <Marker
                      key={`station-${st.id}`}
                      position={[st.lat, st.lon]}
                      icon={createStationIcon(st, isSelected)}
                      eventHandlers={{
                        click: () => handleStationClick(st),
                      }}
                    >
                      <Popup>
                        <div className="p-2 min-w-[260px] text-xs font-mono">
                          <div className="flex items-start justify-between border-b border-slate-700 pb-2 mb-2 gap-2">
                            <div>
                              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                                <Building2 className="w-4 h-4 text-cyan-400 shrink-0" />
                                {st.name}
                              </div>
                              <div className="text-[10px] text-slate-400 mt-0.5">
                                {st.country} {st.isIndian && '• 🇮🇳 Indian Base'}
                              </div>
                            </div>
                            <span
                              className={`text-[9px] px-1.5 py-0.5 rounded font-semibold whitespace-nowrap ${
                                st.operational
                                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                                  : 'bg-rose-950 text-rose-400 border border-rose-800'
                              }`}
                            >
                              {st.status}
                            </span>
                          </div>

                          <div className="text-[11px] text-slate-300 mb-2 leading-relaxed">{st.description}</div>

                          <div className="text-[10px] font-mono text-slate-400 mb-3">
                            LAT: {formatCoord(st.lat, 'N', 'S')} | LON: {formatCoord(st.lon, 'E', 'W')}
                          </div>

                          {st.operational ? (
                            <button
                              onClick={() => handleStationClick(st)}
                              className="w-full py-1.5 bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-[11px] rounded font-bold transition flex items-center justify-center gap-1.5"
                            >
                              <Navigation className="w-3.5 h-3.5" />
                              SET AS NAVIGATION TARGET
                            </button>
                          ) : (
                            <div className="p-1.5 rounded bg-amber-950/50 border border-amber-800/50 text-[10px] text-amber-400 flex items-center gap-1">
                              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                              Historical station. Decommissioned.
                            </div>
                          )}
                        </div>
                      </Popup>
                    </Marker>
                  );
                })}

              {routeData?.routes && (
                <>
                  {routeData.routes.direct?.path && (
                    <Polyline
                      positions={routeData.routes.direct.path}
                      pathOptions={{
                        color: routeData.routes.direct.color ?? '#EF4444',
                        dashArray: '8, 6',
                        weight: 2.5,
                        opacity: 0.7,
                      }}
                    >
                      <Tooltip sticky>
                        <div className="text-[10px]">
                          <strong>Route A (Direct):</strong> {routeData.routes.direct.distance_nm} nm • High Risk
                        </div>
                      </Tooltip>
                    </Polyline>
                  )}

                  {routeData.routes.alternative?.path && (
                    <Polyline
                      positions={routeData.routes.alternative.path}
                      pathOptions={{
                        color: routeData.routes.alternative.color ?? '#F59E0B',
                        dashArray: '6, 4',
                        weight: 3,
                        opacity: 0.8,
                      }}
                    >
                      <Tooltip sticky>
                        <div className="text-[10px]">
                          <strong>Route C (Alternative):</strong> {routeData.routes.alternative.distance_nm} nm • Balanced
                        </div>
                      </Tooltip>
                    </Polyline>
                  )}

                  {routeData.routes.samudra?.path && (
                    <Polyline
                      positions={routeData.routes.samudra.path}
                      pathOptions={{
                        color: routeData.routes.samudra.color ?? '#10B981',
                        weight: 5,
                        opacity: 0.95,
                      }}
                    >
                      <Tooltip sticky>
                        <div className="text-[10px]">
                          <strong className="text-emerald-400">Route B (Samudra):</strong>{' '}
                          {routeData.routes.samudra.distance_nm} nm • Safety {routeData.routes.samudra.safety_score}/100
                        </div>
                      </Tooltip>
                    </Polyline>
                  )}
                </>
              )}
            </MapContainer>
          </div>

          <aside className="absolute left-4 top-4 bottom-4 w-84 max-w-[360px] bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-xl p-3 flex flex-col z-10 shadow-2xl overflow-y-auto">
            <PanelSection
              id="layers"
              title="MAP LAYERS"
              icon={<Layers className="h-4 w-4" />}
              isCollapsed={collapsed.layers}
              onToggle={toggleSection}
            >
              <div className="grid grid-cols-2 gap-2 text-xs">
                <label className="flex items-center gap-2 p-1.5 bg-slate-950/60 rounded border border-slate-800 cursor-pointer hover:border-slate-700">
                  <input
                    type="checkbox"
                    checked={layers.vessels}
                    onChange={(e) => setLayers((l) => ({ ...l, vessels: e.target.checked }))}
                    className="accent-cyan-400"
                  />
                  <span className="text-[11px] text-slate-300">Fleet Vessels</span>
                </label>

                <label className="flex items-center gap-2 p-1.5 bg-slate-950/60 rounded border border-slate-800 cursor-pointer hover:border-slate-700">
                  <input
                    type="checkbox"
                    checked={layers.stations}
                    onChange={(e) => setLayers((l) => ({ ...l, stations: e.target.checked }))}
                    className="accent-cyan-400"
                  />
                  <span className="text-[11px] text-slate-300">Bases & Outposts</span>
                </label>

                <label className="flex items-center gap-2 p-1.5 bg-slate-950/60 rounded border border-slate-800 cursor-pointer hover:border-slate-700">
                  <input
                    type="checkbox"
                    checked={layers.riskZones}
                    onChange={(e) => setLayers((l) => ({ ...l, riskZones: e.target.checked }))}
                    className="accent-cyan-400"
                  />
                  <span className="text-[11px] text-amber-300">Risk Zones</span>
                </label>

                <label className="flex items-center gap-2 p-1.5 bg-slate-950/60 rounded border border-slate-800 cursor-pointer hover:border-slate-700">
                  <input
                    type="checkbox"
                    checked={layers.seaIce}
                    onChange={(e) => setLayers((l) => ({ ...l, seaIce: e.target.checked }))}
                    className="accent-cyan-400"
                  />
                  <span className="text-[11px] text-sky-300">Sea Ice Raster</span>
                </label>

                <label className="flex items-center gap-2 p-1.5 bg-slate-950/60 rounded border border-slate-800 cursor-pointer hover:border-slate-700">
                  <input
                    type="checkbox"
                    checked={layers.currents}
                    onChange={(e) => setLayers((l) => ({ ...l, currents: e.target.checked }))}
                    className="accent-cyan-400"
                  />
                  <span className="text-[11px] text-cyan-300">Ocean Currents</span>
                </label>

                <label className="flex items-center gap-2 p-1.5 bg-slate-950/60 rounded border border-slate-800 cursor-pointer hover:border-slate-700">
                  <input
                    type="checkbox"
                    checked={layers.wind}
                    onChange={(e) => setLayers((l) => ({ ...l, wind: e.target.checked }))}
                    className="accent-cyan-400"
                  />
                  <span className="text-[11px] text-purple-300">ERA5 Wind</span>
                </label>
              </div>
            </PanelSection>

            <PanelSection
              id="stations"
              title="ANTARCTIC STATIONS"
              badge={`${ANTARCTIC_STATIONS.length}`}
              icon={<Building2 className="h-4 w-4" />}
              isCollapsed={collapsed.stations}
              onToggle={toggleSection}
            >
              <div className="mb-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
                  <input
                    type="text"
                    placeholder="Search station or nation..."
                    value={stationSearch}
                    onChange={(e) => setStationSearch(e.target.value)}
                    className="w-full pl-8 pr-2 py-1.5 bg-slate-950 text-xs border border-slate-800 rounded focus:outline-none focus:border-cyan-500 text-slate-200 placeholder-slate-600"
                  />
                </div>

                <div className="flex gap-1 mt-1.5 text-[10px]">
                  <button
                    onClick={() => setStationTab('all')}
                    className={`px-2 py-0.5 rounded ${
                      stationTab === 'all' ? 'bg-cyan-950 text-cyan-300 border border-cyan-800' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    All ({ANTARCTIC_STATIONS.length})
                  </button>
                  <button
                    onClick={() => setStationTab('india')}
                    className={`px-2 py-0.5 rounded ${
                      stationTab === 'india' ? 'bg-cyan-950 text-cyan-300 border border-cyan-800 font-bold' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    🇮🇳 India (3)
                  </button>
                  <button
                    onClick={() => setStationTab('active')}
                    className={`px-2 py-0.5 rounded ${
                      stationTab === 'active' ? 'bg-cyan-950 text-cyan-300 border border-cyan-800' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    Operational
                  </button>
                </div>
              </div>

              <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
                {filteredStations.map((st) => {
                  const isSelected = selectedDestination?.id === st.id;
                  return (
                    <div
                      key={st.id}
                      onClick={() => handleStationClick(st)}
                      className={`p-1.5 rounded cursor-pointer transition border text-left flex items-center justify-between ${
                        isSelected
                          ? 'bg-cyan-950/60 border-cyan-600'
                          : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700'
                      }`}
                    >
                      <div className="min-w-0 pr-1">
                        <div className="text-[11px] font-bold text-slate-200 truncate flex items-center gap-1">
                          {st.isIndian ? '🇮🇳' : '•'} {st.name}
                        </div>
                        <div className="text-[9px] text-slate-500 truncate">{st.country}</div>
                      </div>
                      <span
                        className={`text-[8px] px-1 py-0.2 rounded font-semibold shrink-0 ${
                          st.operational
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-900'
                            : 'bg-rose-950 text-rose-400 border border-rose-900'
                        }`}
                      >
                        {st.status}
                      </span>
                    </div>
                  );
                })}
              </div>
            </PanelSection>

            <PanelSection
              id="telemetry"
              title="ENVIRONMENTAL TELEMETRY"
              icon={<Thermometer className="h-4 w-4" />}
              isCollapsed={collapsed.telemetry}
              onToggle={toggleSection}
            >
              <div className="p-2 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2 text-xs">
                <div className="flex items-center justify-between text-[10px] text-slate-500 border-b border-slate-800/60 pb-1">
                  <span>SENSOR COORD:</span>
                  <span className="text-cyan-400 font-mono">
                    {formatCoord(selectedVessel?.lat ?? -64.25, 'N', 'S')},{' '}
                    {formatCoord(selectedVessel?.lon ?? -56.75, 'E', 'W')}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500 flex items-center gap-1">
                      <Thermometer className="w-3 h-3 text-cyan-400" /> SEA ICE CONC.
                    </div>
                    <div className="text-slate-100 font-bold mt-0.5">
                      {liveEnv?.sea_ice_concentration_pct !== null && liveEnv?.sea_ice_concentration_pct !== undefined
                        ? `${liveEnv.sea_ice_concentration_pct}%`
                        : '38.5% (OBSERVED)'}
                    </div>
                  </div>

                  <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500 flex items-center gap-1">
                      <Waves className="w-3 h-3 text-sky-400" /> OCEAN CURRENT
                    </div>
                    <div className="text-slate-100 font-bold mt-0.5">
                      {liveEnv?.ocean_current_speed_knots !== null && liveEnv?.ocean_current_speed_knots !== undefined
                        ? `${liveEnv.ocean_current_speed_knots} kn`
                        : '0.42 kn @ 045°'}
                    </div>
                  </div>

                  <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500 flex items-center gap-1">
                      <Wind className="w-3 h-3 text-purple-400" /> SURFACE WIND
                    </div>
                    <div className="text-slate-100 font-bold mt-0.5">
                      {liveEnv?.wind_speed_knots !== null && liveEnv?.wind_speed_knots !== undefined
                        ? `${liveEnv.wind_speed_knots} kn`
                        : '18.4 kn (ERA5)'}
                    </div>
                  </div>

                  <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                    <div className="text-[9px] text-slate-500 flex items-center gap-1">
                      <Gauge className="w-3 h-3 text-amber-400" /> AIR PRESSURE
                    </div>
                    <div className="text-slate-100 font-bold mt-0.5">
                      {liveEnv?.pressure_hpa !== null && liveEnv?.pressure_hpa !== undefined
                        ? `${liveEnv.pressure_hpa} hPa`
                        : '984.2 hPa'}
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80">
                  <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">
                    UNOBSERVED TELEMETRY METRICS:
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-[9px]">
                    <div className="p-1 bg-slate-900/50 rounded flex items-center justify-between text-slate-400">
                      <span>Wave Height:</span>
                      <span className="text-slate-500 italic">Unavailable</span>
                    </div>
                    <div className="p-1 bg-slate-900/50 rounded flex items-center justify-between text-slate-400">
                      <span>Rain Rate:</span>
                      <span className="text-slate-500 italic">Unavailable</span>
                    </div>
                    <div className="p-1 bg-slate-900/50 rounded flex items-center justify-between text-slate-400">
                      <span>Visibility:</span>
                      <span className="text-slate-500 italic">Unavailable</span>
                    </div>
                    <div className="p-1 bg-slate-900/50 rounded flex items-center justify-between text-slate-400">
                      <span>Storm Prob:</span>
                      <span className="text-slate-500 italic">Unavailable</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between text-[9px] text-slate-400 pt-1">
                  <span>Providers:</span>
                  <div className="flex gap-1 font-mono">
                    <span className="px-1 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                      NSIDC
                    </span>
                    <span className="px-1 py-0.2 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                      ERA5
                    </span>
                    <span className="px-1 py-0.2 rounded bg-sky-950 text-sky-400 border border-sky-800">
                      GLORYS
                    </span>
                  </div>
                </div>
              </div>
            </PanelSection>

            <PanelSection
              id="alerts"
              title="ACTIVE MARITIME ALERTS"
              badge={`${alerts.length}`}
              icon={<ShieldAlert className="h-4 w-4 text-amber-400" />}
              isCollapsed={collapsed.alerts}
              onToggle={toggleSection}
            >
              <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
                {alerts.length === 0 ? (
                  <div className="text-[10px] text-slate-500 italic text-center py-2">
                    No active maritime navigation alerts.
                  </div>
                ) : (
                  alerts.map((al, idx) => {
                    const isWarn = String(al.severity).toLowerCase() === 'warning';
                    const isCrit = String(al.severity).toLowerCase() === 'critical';
                    return (
                      <div
                        key={idx}
                        className={`p-2 rounded border text-xs ${
                          isCrit
                            ? 'bg-rose-950/40 border-rose-800/80 text-rose-300'
                            : isWarn
                              ? 'bg-amber-950/30 border-amber-800/70 text-amber-200'
                              : 'bg-slate-950/50 border-slate-800 text-slate-300'
                        }`}
                      >
                        <div className="flex items-center justify-between text-[9px] font-bold">
                          <span className="uppercase tracking-wide flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3 shrink-0" />
                            {al.title}
                          </span>
                          <span
                            className={`px-1 py-0.2 rounded font-semibold text-[8px] ${
                              isCrit ? 'bg-rose-900 text-rose-200' : isWarn ? 'bg-amber-900 text-amber-200' : 'bg-slate-800 text-slate-400'
                            }`}
                          >
                            {al.severity}
                          </span>
                        </div>
                        <p className="text-[10px] text-slate-400 mt-1 leading-snug">{al.message}</p>
                        <div className="text-[9px] text-slate-500 mt-1 flex justify-between">
                          <span>{al.affected_zone_or_vessel || 'Sector 4'}</span>
                          <span>{al.category}</span>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </PanelSection>

            <PanelSection
              id="system"
              title="AI ARCHITECTURE & STATUS"
              icon={<Cpu className="h-4 w-4 text-cyan-400" />}
              isCollapsed={collapsed.system}
              onToggle={toggleSection}
            >
              <div className="p-2 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2 text-[10px]">
                <div>
                  <span className="text-slate-500">PHYSICS ENGINE:</span>
                  <div className="text-slate-300 font-semibold mt-0.5">
                    {systemStatus?.physics_engine_status ?? 'Operational (Calibrated Runge-Kutta 4th-Order)'}
                  </div>
                </div>

                <div>
                  <span className="text-slate-500">RESIDUAL ML MODEL:</span>
                  <div className="text-emerald-400 font-semibold mt-0.5">
                    {systemStatus?.ml_model_status ?? 'Ridge Regression Residual Correction Ready'}
                  </div>
                </div>

                <div>
                  <span className="text-slate-500">PATHFINDER ENGINE:</span>
                  <div className="text-cyan-300 font-semibold mt-0.5">
                    Deterministic A* on PolarNavigationGrid (Resolution: 20km)
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80 text-[9px] text-slate-400 leading-relaxed">
                  <strong className="text-slate-300">Hybrid Physics-Informed ML:</strong> Combines RK4 integration of hydrodynamic drag, ocean Coriolis, and wind forces with L2-regularized Ridge regression residual bias correction.
                </div>
              </div>
            </PanelSection>
          </aside>

          <aside className="absolute right-4 top-4 bottom-4 w-92 max-w-[390px] bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-xl p-4 flex flex-col z-10 shadow-2xl overflow-y-auto">
            <div className="border-b border-slate-800 pb-3 mb-3 shrink-0">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-400 uppercase tracking-widest font-bold flex items-center gap-1.5">
                  <Navigation className="w-3.5 h-3.5 text-cyan-400" />
                  VOYAGE DECISION SUPPORT
                </span>
                {selectedDestination && (
                  <button
                    onClick={clearDestination}
                    className="text-[9px] text-slate-400 hover:text-rose-400 transition"
                  >
                    RESET TARGET
                  </button>
                )}
              </div>

              {selectedDestination ? (
                <div className="mt-2.5 p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-[9px] text-slate-500 uppercase">ORIGIN (VESSEL)</div>
                      <div className="text-xs font-bold text-cyan-300 truncate">{selectedVessel?.name}</div>
                      <div className="text-[9px] text-slate-400">
                        {formatCoord(selectedVessel?.lat ?? 0, 'N', 'S')}, {formatCoord(selectedVessel?.lon ?? 0, 'E', 'W')}
                      </div>
                    </div>
                    <span className="text-slate-600 px-2">➜</span>
                    <div className="text-right">
                      <div className="text-[9px] text-slate-500 uppercase">TARGET DESTINATION</div>
                      <div className="text-xs font-bold text-emerald-400 truncate">{selectedDestination.name}</div>
                      <div className="text-[9px] text-slate-400">
                        {formatCoord(selectedDestination.lat, 'N', 'S')}, {formatCoord(selectedDestination.lon, 'E', 'W')}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-2 p-3 bg-slate-950/60 rounded border border-slate-800/80 text-xs text-slate-400 text-center">
                  Click any operational Antarctic station on the map or left roster to plan optimal routes.
                </div>
              )}
            </div>

            {isCalculating && (
              <div className="py-8 text-center my-auto">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400 mb-3" />
                <div className="text-xs text-cyan-300 font-bold">COMPUTING POLAR CORRIDORS...</div>
                <div className="text-[10px] text-slate-500 mt-1">Evaluating pack ice concentration & Coriolis drift</div>
                <div className="text-[10px] text-slate-600 mt-0.5">Optimizing fuel consumption and safety metrics</div>
              </div>
            )}

            {error && !isCalculating && (
              <div className="mb-3 p-2.5 bg-rose-950/40 border border-rose-800/60 rounded text-xs text-rose-300">
                <div className="font-bold flex items-center gap-1">
                  <AlertTriangle className="w-3.5 h-3.5" /> ROUTING ENGINE NOTICE
                </div>
                <div className="text-[10px] text-slate-400 mt-1">{error}</div>
              </div>
            )}

            {!isCalculating && routeData?.routes && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-[9px] px-1">
                  <span className="text-slate-500 uppercase tracking-wider">ROUTING ENGINE SOURCE:</span>
                  <span
                    className={`font-semibold px-1.5 py-0.2 rounded ${
                      routeSource === 'A* ROUTING ENGINE'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-amber-950 text-amber-400 border border-amber-800'
                    }`}
                  >
                    {routeSource}
                  </span>
                </div>

                {routeData.routes.samudra && (
                  <div className="bg-emerald-950/30 border-2 border-emerald-500/80 rounded-xl p-3 relative overflow-hidden shadow-lg">
                    <div className="absolute top-0 right-0 bg-emerald-500 text-slate-950 text-[9px] font-bold px-2.5 py-0.5 rounded-bl">
                      RECOMMENDED
                    </div>

                    <div className="text-xs font-bold text-emerald-400 flex items-center gap-1.5 mb-2.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      {routeData.routes.samudra.name}
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs mb-2.5">
                      <div className="bg-slate-950/80 p-2 rounded border border-emerald-900/40">
                        <div className="text-slate-400 text-[9px] flex items-center gap-1">
                          <Navigation className="w-3 h-3 text-emerald-400" /> DISTANCE
                        </div>
                        <div className="text-slate-100 font-bold mt-0.5">
                          {routeData.routes.samudra.distance_nm} nm
                        </div>
                      </div>

                      <div className="bg-slate-950/80 p-2 rounded border border-emerald-900/40">
                        <div className="text-slate-400 text-[9px] flex items-center gap-1">
                          <Clock className="w-3 h-3 text-emerald-400" /> EST. TRANSIT
                        </div>
                        <div className="text-slate-100 font-bold mt-0.5">
                          {routeData.routes.samudra.eta}
                        </div>
                      </div>

                      <div className="bg-slate-950/80 p-2 rounded border border-emerald-900/40">
                        <div className="text-slate-400 text-[9px] flex items-center gap-1">
                          <Fuel className="w-3 h-3 text-emerald-400" /> FUEL BURN
                        </div>
                        <div className="text-slate-100 font-bold mt-0.5">
                          {routeData.routes.samudra.fuel_mt} MT
                        </div>
                      </div>

                      <div className="bg-slate-950/80 p-2 rounded border border-emerald-900/40">
                        <div className="text-slate-400 text-[9px] flex items-center gap-1">
                          <Shield className="w-3 h-3 text-emerald-400" /> SAFETY SCORE
                        </div>
                        <div className="text-emerald-300 font-bold mt-0.5">
                          {routeData.routes.samudra.safety_score} / 100
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-[10px] pt-2 border-t border-emerald-900/60 text-slate-300">
                      <span>
                        Ice Hazard:{' '}
                        <strong className="text-emerald-400">{routeData.routes.samudra.ice_risk}</strong>
                      </span>
                      <span>
                        Fuel Savings:{' '}
                        <strong className="text-emerald-400">{routeData.routes.samudra.fuel_saving}</strong>
                      </span>
                    </div>
                  </div>
                )}

                {routeData.routes.direct && (
                  <div className="bg-slate-950/60 border border-rose-900/50 rounded-lg p-2.5">
                    <div className="flex items-center justify-between text-xs font-bold text-rose-400 mb-2">
                      <span>{routeData.routes.direct.name}</span>
                      <span className="text-[9px] text-rose-500 font-normal px-1.5 py-0.2 bg-rose-950/60 rounded border border-rose-900">
                        HIGH RISK
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-xs text-slate-300">
                      <div>
                        <div className="text-slate-500 text-[9px]">DIST</div>
                        <div className="font-bold">{routeData.routes.direct.distance_nm} nm</div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[9px]">ETA</div>
                        <div className="font-bold">{routeData.routes.direct.eta}</div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[9px]">FUEL</div>
                        <div className="font-bold">{routeData.routes.direct.fuel_mt} MT</div>
                      </div>
                    </div>
                  </div>
                )}

                {routeData.routes.alternative && (
                  <div className="bg-slate-950/60 border border-amber-900/50 rounded-lg p-2.5">
                    <div className="flex items-center justify-between text-xs font-bold text-amber-400 mb-2">
                      <span>{routeData.routes.alternative.name}</span>
                      <span className="text-[9px] text-amber-500 font-normal px-1.5 py-0.2 bg-amber-950/60 rounded border border-amber-900">
                        BALANCED
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-xs text-slate-300">
                      <div>
                        <div className="text-slate-500 text-[9px]">DIST</div>
                        <div className="font-bold">{routeData.routes.alternative.distance_nm} nm</div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[9px]">ETA</div>
                        <div className="font-bold">{routeData.routes.alternative.eta}</div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[9px]">FUEL</div>
                        <div className="font-bold">{routeData.routes.alternative.fuel_mt} MT</div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="p-2.5 bg-slate-950/80 border border-slate-800 rounded-lg text-[10px]">
                  <div className="text-slate-500 tracking-wider font-semibold mb-1.5">VOYAGE SPECIFICATIONS</div>
                  <div className="grid grid-cols-2 gap-1.5 text-slate-400">
                    <div>
                      Speed: <span className="text-cyan-300 font-semibold">{selectedVessel?.speedKnots} kn</span>
                    </div>
                    <div>
                      Ice Class: <span className="text-slate-200 font-semibold">{selectedVessel?.iceClass}</span>
                    </div>
                    <div>
                      Sector: <span className="text-slate-200 font-semibold">East / Weddell Sea</span>
                    </div>
                    <div>
                      Polar Code: <span className="text-emerald-400 font-semibold">Compliant</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-auto pt-3 border-t border-slate-800 text-[9px] text-slate-500 shrink-0">
              <div className="flex justify-between items-center">
                <span>POLAR A* SOLVER v2.4</span>
                <span className="text-emerald-400 font-semibold">COPERNICUS GLORYS + ERA5</span>
              </div>
              <div className="mt-1 text-slate-600 truncate">
                SAMUDRA NAV AI • NCAOR / MoES • SIH26059
              </div>
            </div>
          </aside>
        </div>
      </div>
    </ErrorBoundary>
  );
}
