import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mock helpers

def create_mock_service(is_loaded=True):
    """
    Returns a MagicMock whose attributes and methods mirror the
    classifier_service singleton without touching PyTorch nor the BERT
    checkpoint.

    :param is_loaded: simulates whether the model loaded successfully
    :return: configured MagicMock instance
    """
    mock = MagicMock()

    mock.is_loaded = is_loaded
    mock.model_load_time = 0.52 if is_loaded else None
    mock.model_load_error = None if is_loaded else "File not found"
    mock.load_model.return_value = is_loaded

    mock.predict.return_value = {
        "predicted_class": "Business",
        "confidence": 0.9852,
        "all_probabilities": {
            "World": 0.0002,
            "Sports": 5.4409e-05,
            "Business": 0.9852,
            "Science": 0.0144,
        },
        "sorted_predictions": [
            ("Business", 0.9852),
            ("Science", 0.0144),
            ("World", 0.0002),
            ("Sports", 5.4409e-05),
        ],
        "input_text": "The stock market surged today as investors reacted to stronger than expected earnings reports from m...",
    }

    return mock

# Fixtures

@pytest.fixture
def client():
    """TestClient backed by a mock classifier service (model loaded)."""
    mock_service = create_mock_service()
    with patch("app.main.classifier_service", mock_service):
        from app.main import app
        with TestClient(app) as c:
            yield c

@pytest.fixture
def unloaded_client():
    """TestClient backed by a mock classifier service (model NOT loaded)."""
    mock_service = create_mock_service(is_loaded=False)
    with patch("app.main.classifier_service", mock_service):
        from app.main import app
        with TestClient(app) as c:
            yield c

# GET /

class TestRoot:
    """Tests for the root endpoint"""

    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_contains_expected_keys(self, client):
        data = client.get("/").json()
        assert "message" in data
        assert "status" in data
        assert "model_loaded" in data
        assert "model_load_time" in data
        assert "endpoints" in data

    def test_status_loaded_model(self, client):
        data = client.get("/").json()
        assert data["model_loaded"] is True
        assert data["status"] == "MODEL LOADED"

    def test_status_unloaded_model(self, unloaded_client):
        data = unloaded_client.get("/").json()
        assert data["model_loaded"] is False
        assert data["status"] == "MODEL NOT LOADED"

# GET /health

class TestHealth:
    """Tests for the health diagnostic endpoint"""

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_healthy_when_loaded(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_unhealthy_when_not_loaded(self, unloaded_client):
        data = unloaded_client.get("/health").json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False

# POST /predict - valid

class TestPredictValid:
    """Tests for successful predictions conforming to PredictionResponse."""

    def test_returns_200(self, client):
        response = client.post("/predict", json={"text": "The stock market surged today."})
        assert response.status_code == 200

    def test_response_has_all_fields(self, client):
        data = client.post(
            "/predict", json={"text": "Apple unveiled its latest iPhone model today."}
        ).json()

        expected_fields = {"text", "category", "confidence", "all_probabilities", "processing_time"}

        assert expected_fields.issubset(data.keys())

    def test_response_field_types(self, client):
        data = client.post(
            "/predict", json={"text": "NASA launched a new rover to Mars."}
        ).json()

        assert isinstance(data["text"], str)
        assert isinstance(data["category"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["all_probabilities"], dict)
        assert isinstance(data["processing_time"], (int, float))

    def test_valid_category(self, client):
        data = client.post(
            "/predict",
            json={"text": "Scientists discovered a new deep-sea species."},
        ).json()

        valid_categories = {"World", "Sports", "Business", "Science"}
        assert data["category"] in valid_categories

    def test_confidence_between_zero_and_one(self, client):
        data = client.post(
            "/predict",
            json={"text": "The Lakers won the championship last night."},
        ).json()

        assert 0.0 <= data["confidence"] <= 1.0

    def test_all_probabilities_contains_four_classes(self, client):
        data = client.post(
            "/predict",
            json={"text": "Global trade tensions escalated this week."},
        ).json()

        expected_classes = {"World", "Sports", "Business", "Science"}
        assert set(data["all_probabilities"].keys()) == expected_classes

# POST /predict - validation errors (422)

class TestPredictValidation:
    """Tests for Pydantic input validation on TextRequest."""

    def test_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_missing_text_field_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_text_exceeding_10000_chars_returns_422(self, client):
        response = client.post("/predict", json={"text": "a" * 10001})
        assert response.status_code == 422

    def test_text_at_exact_limit_returns_200(self, client):
        response = client.post("/predict", json={"text": "a" * 10000})
        assert response.status_code == 200

    def test_no_json_body_returns_422(self, client):
        response = client.post("/predict")
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client):
        response = client.post("/predict", json={"text": 54321})
        assert response.status_code == 422

    def test_422_body_contains_detail(self, client):
        data = client.post("/predict", json={"text": ""}).json()
        assert "detail" in data

# POST /predict - model not loaded (503)

class TestPredictServiceUnavailable:
    """ Tests for when the classifier is not loaded."""

    def test_returns_503_when_model_not_loaded(self, unloaded_client):
        response = unloaded_client.post("/predict", json={"text": "Totally valid text"})
        assert response.status_code == 503

    def test_503_body_contains_detail(self, unloaded_client):
        data = unloaded_client.post("/predict", json={"text": "Totally valid text"}).json()
        assert "detail" in data
