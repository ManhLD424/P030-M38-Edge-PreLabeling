"""CVAT REST API Client for 3D LiDAR Annotation Tasks and Jobs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

from ..geometry.cuboid import Cuboid3D
from ..geometry.cvat_adapter import CVAT3DAdapter
from ..pointcloud.pcd_io import read_pcd, PCDData


class CVAT3DClient:
    """Handles communication with CVAT 3D server REST API."""

    def __init__(
        self,
        base_url: str = "https://cvat.note.transformerlabs.ai",
        token: Optional[str] = None,
        default_org: Optional[str] = "ai20k-cohort-4a",
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.default_org = default_org
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    @classmethod
    def from_config_or_file(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        token_txt_path: Optional[Union[str, Path]] = r"d:\Project\VinPrj\CVAT.txt"
    ) -> CVAT3DClient:
        """Initializes client by reading token from config file or CVAT.txt."""
        base_url = "https://cvat.note.transformerlabs.ai"
        token = None
        org = "ai20k-cohort-4a"

        # Check CVAT.txt if exists
        p_txt = Path(token_txt_path) if token_txt_path else None
        if p_txt and p_txt.is_file():
            text = p_txt.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in ("online", "cvat"):
                        token = v
                        break

        # Check yaml config
        if config_path:
            p_cfg = Path(config_path)
            if p_cfg.is_file():
                try:
                    import yaml
                    with p_cfg.open("r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    server = data.get("server", {})
                    base_url = server.get("url", base_url)
                    org = server.get("organization", org)
                    if not token:
                        token = server.get("token")
                except Exception:
                    pass

        return cls(base_url=base_url, token=token, default_org=org)

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/{path.lstrip('/')}"

    def _headers(self, org: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        use_org = org if org is not None else self.default_org
        if use_org:
            headers["X-Organization"] = use_org
        return headers

    def ping(self) -> Tuple[bool, str]:
        """Checks connection to CVAT server."""
        if not self.token:
            return False, "Chưa có CVAT token để xác thực"
        try:
            r = self.session.get(self._url("users/self"), timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return True, f"OK (User: {data.get('username')}, ID: {data.get('id')})"
            return False, f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as e:
            return False, str(e)

    def get_job_info(self, job_id: int) -> Dict[str, Any]:
        """Fetches metadata for a single Job."""
        r = self.session.get(self._url(f"jobs/{job_id}"), headers=self._headers(), timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Lỗi lấy thông tin Job {job_id}: {r.status_code} {r.text[:120]}")
        return r.json()

    def get_labels(self, job_id: int) -> List[Dict[str, Any]]:
        """Fetches label ontology for a Job."""
        r = self.session.get(self._url(f"labels?job_id={job_id}"), headers=self._headers(), timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Lỗi lấy danh mục nhãn Job {job_id}: {r.status_code}")
        data = r.json()
        return data.get("results", data if isinstance(data, list) else [])

    def get_annotations(self, job_id: int) -> Dict[str, Any]:
        """Fetches current annotations for a Job."""
        r = self.session.get(self._url(f"jobs/{job_id}/annotations"), headers=self._headers(), timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Lỗi lấy annotations Job {job_id}: {r.status_code}")
        return r.json()

    def save_annotations(
        self,
        job_id: int,
        payload: Dict[str, Any],
        mode: str = "merge"
    ) -> Dict[str, Any]:
        """Uploads annotations to Job (mode: 'merge' or 'replace')."""
        if mode == "replace":
            # PUT replaces all annotations
            r = self.session.put(
                self._url(f"jobs/{job_id}/annotations"),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout
            )
        else:
            # PATCH action=update merges new shapes
            r = self.session.patch(
                self._url(f"jobs/{job_id}/annotations?action=update"),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout
            )
        if r.status_code not in (200, 201, 202):
            raise RuntimeError(f"Lỗi upload annotations Job {job_id}: {r.status_code} {r.text[:200]}")
        return r.json()

    def download_frame_pcd(self, job_id: int, frame_idx: int, save_path: Union[str, Path]) -> Path:
        """Downloads point cloud file for a frame in Job."""
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        r = self.session.get(
            self._url(f"jobs/{job_id}/data?type=frame&number={frame_idx}"),
            headers=self._headers(),
            timeout=self.timeout
        )
        if r.status_code != 200:
            raise RuntimeError(f"Lỗi tải point cloud frame {frame_idx} của Job {job_id}: {r.status_code}")
        p.write_bytes(r.content)
        return p
