
from fastapi import APIRouter, HTTPException

router = APIRouter()

users = {
    "admin": {"role": "admin"},
    "user1": {"role": "user"},
}
roles_permissions = {
    "admin": ["read", "write", "delete", "manage"],
    "user": ["read", "write"],
}

def get_current_user(username: str) -> dict:
    if username not in users:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return users[username]

@router.get("/rbac/permissions/{username}")
def get_permissions(username: str) -> dict:
    user = get_current_user(username)
    role = user["role"]
    return {"role": role, "permissions": roles_permissions.get(role, [])}

@router.post("/rbac/add_user")
def add_user(username: str, role: str) -> dict:
    if role not in roles_permissions:
        raise HTTPException(status_code=400, detail="Invalid role")
    users[username] = {"role": role}
    return {"status": "added", "username": username, "role": role}
