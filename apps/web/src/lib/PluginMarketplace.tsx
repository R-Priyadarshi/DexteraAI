import React from 'react';

const plugins = [
  {
    name: 'Gesture Plugin',
    description: 'Add custom gesture recognition algorithms.',
    installed: false,
  },
  {
    name: 'Analytics Plugin',
    description: 'Real-time analytics and dashboard integration.',
    installed: false,
  },
  {
    name: 'Edge Sync Plugin',
    description: 'Sync models and data across edge devices.',
    installed: false,
  },
];

export default function PluginMarketplace() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-6">DexteraAI Plugin Marketplace</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {plugins.map((plugin, idx) => (
          <div key={idx} className="border rounded-lg p-4 shadow bg-white">
            <h2 className="text-xl font-semibold mb-2">{plugin.name}</h2>
            <p className="mb-4 text-gray-700">{plugin.description}</p>
            <button className={`px-4 py-2 rounded ${plugin.installed ? 'bg-green-500 text-white' : 'bg-blue-500 text-white'}`}
              disabled={plugin.installed}
            >
              {plugin.installed ? 'Installed' : 'Install'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
