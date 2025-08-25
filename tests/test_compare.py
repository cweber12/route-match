# tests/test_compare_image.py
import pytest
from httpx import AsyncClient
from fastapi import UploadFile
from main import app
from io import BytesIO

# --------------------------------------------------------------------------
# Test endpoint that compares the uploaded image against stored climbing data
# and generates a video.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_compare():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        fake_image = BytesIO(b"test image content")
        files = {
            "image": ("test.jpg", fake_image, "image/jpeg"),
            "built_in_image": ("", BytesIO(), "application/octet-stream"),  # Empty if using upload
        }

        data = {
            "pose_lm_in": "",
            "sift_kp_in": "",
            "preprocess": "true",
            "sift_left": "20",
            "sift_right": "20",
            "sift_up": "20",
            "sift_down": "20",
            "line_color": "0,255,0",
            "point_color": "255,0,0",
            "s3_folders": "Cole/Yosemite/ElCapitan/2025-07-01T00:00:00",  # Can pass multiple like this
        }

        response = await ac.post("/api/compare-image", files=files, data=data)
        assert response.status_code == 200
        assert "video_url" in response.json()

# --------------------------------------------------------------------------
# Test different combinations of parameters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("preprocess,sift_params,colors,expected_status", [
    # Test basic valid parameters
    ("true", {"sift_left": "20", "sift_right": "20", "sift_up": "20", "sift_down": "20"}, 
     {"line_color": "0,255,0", "point_color": "255,0,0"}, 200),
    
    # Test extreme SIFT boundaries
    ("false", {"sift_left": "0", "sift_right": "0", "sift_up": "0", "sift_down": "0"}, 
     {"line_color": "100,100,255", "point_color": "255,100,100"}, 200),
    
    # Test maximum SIFT coverage
    ("true", {"sift_left": "50", "sift_right": "50", "sift_up": "50", "sift_down": "50"}, 
     {"line_color": "255,255,0", "point_color": "0,255,255"}, 200),
    
    # Test negative SIFT values (should still work)
    ("false", {"sift_left": "-10", "sift_right": "-5", "sift_up": "10", "sift_down": "15"}, 
     {"line_color": "255,0,128", "point_color": "128,255,0"}, 200),
    
    # Test decimal SIFT values
    ("true", {"sift_left": "12.5", "sift_right": "17.8", "sift_up": "22.3", "sift_down": "8.9"}, 
     {"line_color": "200,150,100", "point_color": "50,100,200"}, 200),
])

@pytest.mark.asyncio
async def test_parameter_variations(preprocess, sift_params, colors, expected_status):
    """Test various combinations of client parameters"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        files = {
            "image": ("test_param.jpg", BytesIO(b"test image data"), "image/jpeg"),
            "built_in_image": ("", BytesIO(), "application/octet-stream"),
        }

        data = {
            "image": "https://route-keypoints.s3.amazonaws.com/sample-images/MidnightLightning.jpg",
            "pose_lm_in": "",
            "sift_kp_in": "",
            "preprocess": preprocess,
            "s3_folders": "Cole/Yosemite/ElCapitan/2025-07-01T00:00:00",
            **sift_params,
            **colors
        }

        response = await ac.post("/api/compare-image", files=files, data=data)
        assert response.status_code == expected_status
        if expected_status == 200:
            assert "video_url" in response.json()

@pytest.mark.asyncio
async def test_missing_required_parameters():
    """Test behavior when required parameters are missing"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        files = {
            "image": ("test.jpg", BytesIO(b"test content"), "image/jpeg"),
            "built_in_image": ("", BytesIO(), "application/octet-stream"),
        }

        # Test missing s3_folders
        data_missing_s3 = {
            "pose_lm_in": "",
            "sift_kp_in": "",
            "preprocess": "true",
            "sift_left": "20",
            "sift_right": "20",
            "sift_up": "20",
            "sift_down": "20",
            "line_color": "0,255,0",
            "point_color": "255,0,0",
            # "s3_folders": missing
        }

        response = await ac.post("/api/compare-image", files=files, data=data_missing_s3)
        # Should either return 422 (validation error) or 200 with error handling
        assert response.status_code in [200, 422]


@pytest.mark.asyncio
async def test_extreme_parameter_values():
    """Test extreme parameter values to check robustness"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        files = {
            "image": ("extreme_test.jpg", BytesIO(b"extreme test data"), "image/jpeg"),
            "built_in_image": ("", BytesIO(), "application/octet-stream"),
        }

        # Test very large SIFT values
        data_extreme = {
            "preprocess": "false",
            "sift_left": "999",
            "sift_right": "999", 
            "sift_up": "999",
            "sift_down": "999",
            "line_color": "255, 255, 255",  
            "point_color": "0, 0, 255",  
            "s3_folders": "Cole/Yosemite/ElCapitan/2025-07-01T00:00:00",
        }

        response = await ac.post("/api/compare-image", files=files, data=data_extreme)
        # Should handle extreme values gracefully
        assert response.status_code == 200