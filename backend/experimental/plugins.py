from fastapi import APIRouter, HTTPException

router = APIRouter()

# In-memory plugin registry for demo purposes
plugin_registry = [
    {
        "name": "Gesture Plugin",
        "description": "Add custom gesture recognition algorithms.",
        "installed": False,
    },
    {
        "name": "Analytics Plugin",
        "description": "Real-time analytics and dashboard integration.",
        "installed": False,
    },
    {
        "name": "Edge Sync Plugin",
        "description": "Sync models and data across edge devices.",
        "installed": False,
    },
]


@router.get("/plugins", response_model=list[dict])
def list_plugins() -> list[dict]:
    return plugin_registry


@router.post("/plugins/install/{plugin_name}")
def install_plugin(plugin_name: str) -> dict:
    for plugin in plugin_registry:
        if plugin["name"] == plugin_name:
            if plugin["installed"]:
                raise HTTPException(status_code=400, detail="Plugin already installed.")
            plugin["installed"] = True
            return {"status": "installed", "plugin": plugin}
    raise HTTPException(status_code=404, detail="Plugin not found.")
