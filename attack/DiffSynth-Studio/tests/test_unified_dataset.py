import pytest
import torch
import json
import pandas as pd
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Import the classes we need to test
import sys
sys.path.append('/data1/humw/Codes/Attack_Unified_Model/attack/DiffSynth-Studio')

from diffsynth.core.data.unified_dataset import UnifiedDataset
from diffsynth.core.data.operators import (
    LoadImage, ImageCropAndResize, ToAbsolutePath, RouteByType, 
    RouteByExtensionName, LoadTorchPickle, ToList, SequencialProcess,
    LoadVideo, LoadGIF, DataProcessingPipeline, DataProcessingOperator
)


class TestUnifiedDataset:
    """Test suite for UnifiedDataset class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON metadata for testing."""
        return [
            {"id": 1, "image": "test1.jpg", "text": "sample text 1"},
            {"id": 2, "image": "test2.png", "text": "sample text 2"},
            {"id": 3, "image": "test3.jpg", "text": "sample text 3"}
        ]
    
    @pytest.fixture
    def sample_jsonl_data(self):
        """Sample JSONL metadata for testing."""
        return [
            {"id": 1, "video": "test1.mp4", "description": "video 1"},
            {"id": 2, "video": "test2.avi", "description": "video 2"}
        ]
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV metadata for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'filename': ['file1.jpg', 'file2.png', 'file3.gif'],
            'label': ['cat', 'dog', 'bird']
        })
    
    @pytest.fixture
    def create_test_files(self, temp_dir, sample_json_data, sample_jsonl_data, sample_csv_data):
        """Create test files in temporary directory."""
        # Create JSON file
        json_path = os.path.join(temp_dir, "test_data.json")
        with open(json_path, 'w') as f:
            json.dump(sample_json_data, f)
        
        # Create JSONL file
        jsonl_path = os.path.join(temp_dir, "test_data.jsonl")
        with open(jsonl_path, 'w') as f:
            for item in sample_jsonl_data:
                f.write(json.dumps(item) + '\n')
        
        # Create CSV file
        csv_path = os.path.join(temp_dir, "test_data.csv")
        sample_csv_data.to_csv(csv_path, index=False)
        
        # Create dummy image files
        for img_name in ['test1.jpg', 'test2.png', 'test3.jpg']:
            img_path = os.path.join(temp_dir, img_name)
            Image.new('RGB', (100, 100), color='red').save(img_path)
        
        # Create dummy cache files
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)
        for i in range(3):
            cache_path = os.path.join(cache_dir, f"cache_{i}.pth")
            torch.save({"data": f"cached_data_{i}"}, cache_path)
        
        return {
            'json': json_path,
            'jsonl': jsonl_path,
            'csv': csv_path,
            'cache_dir': cache_dir
        }
    
    def test_init_default_parameters(self):
        """Test UnifiedDataset initialization with default parameters."""
        dataset = UnifiedDataset()
        assert dataset.base_path is None
        assert dataset.metadata_path is None
        assert dataset.repeat == 1
        assert dataset.data_file_keys == tuple()
        assert dataset.data == []
        assert dataset.cached_data == []
        assert dataset.load_from_cache is True
        assert dataset.special_operator_map == {}
    
    def test_init_with_parameters(self):
        """Test UnifiedDataset initialization with custom parameters."""
        def dummy_operator(x): return x
        dataset = UnifiedDataset(
            base_path="/test/path",
            metadata_path="/test/metadata.json",
            repeat=3,
            data_file_keys=("image", "video"),
            main_data_operator=dummy_operator,
            special_operator_map={"image": dummy_operator}
        )
        assert dataset.base_path == "/test/path"
        assert dataset.metadata_path == "/test/metadata.json"
        assert dataset.repeat == 3
        assert dataset.data_file_keys == ("image", "video")
        assert dataset.main_data_operator == dummy_operator
        assert dataset.special_operator_map == {"image": dummy_operator}
    
    def test_load_metadata_json(self, temp_dir, create_test_files, sample_json_data):
        """Test loading metadata from JSON file."""
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['json']
        )
        assert dataset.data == sample_json_data
        assert not dataset.load_from_cache
    
    def test_load_metadata_jsonl(self, temp_dir, create_test_files, sample_jsonl_data):
        """Test loading metadata from JSONL file."""
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['jsonl']
        )
        assert dataset.data == sample_jsonl_data
        assert not dataset.load_from_cache
    
    def test_load_metadata_csv(self, temp_dir, create_test_files, sample_csv_data):
        """Test loading metadata from CSV file."""
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['csv']
        )
        expected_data = sample_csv_data.to_dict('records')
        assert dataset.data == expected_data
        assert not dataset.load_from_cache
    
    def test_load_metadata_no_metadata_path(self, temp_dir, create_test_files):
        """Test loading without metadata path (search for cache files)."""
        dataset = UnifiedDataset(
            base_path=create_test_files['cache_dir'],
            metadata_path=None
        )
        assert dataset.load_from_cache
        assert len(dataset.cached_data) == 3
        assert all(path.endswith('.pth') for path in dataset.cached_data)
    
    def test_getitem_from_cache(self, temp_dir, create_test_files):
        """Test getting item from cached data."""
        dataset = UnifiedDataset(
            base_path=create_test_files['cache_dir'],
            metadata_path=None
        )
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'data' in item
    
    def test_getitem_from_metadata(self, temp_dir, create_test_files, sample_json_data):
        """Test getting item from metadata."""
        def dummy_operator(x): return x
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['json'],
            data_file_keys=("image",),
            main_data_operator=dummy_operator
        )
        item = dataset[0]
        assert item == sample_json_data[0]
    
    def test_getitem_with_special_operator(self, temp_dir, create_test_files):
        """Test getting item with special operator."""
        def special_op(x): return f"special_{x}"
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['json'],
            data_file_keys=("image",),
            special_operator_map={"image": special_op}
        )
        item = dataset[0]
        assert item['image'] == "special_test1.jpg"
    
    def test_getitem_with_modulo_operator(self, temp_dir, create_test_files, sample_json_data):
        """Test getting item with modulo operation for repeat."""
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['json'],
            repeat=2
        )
        # Should return same item for indices 0 and 3 (3 % 3 = 0)
        item1 = dataset[0]
        item2 = dataset[3]
        assert item1 == item2 == sample_json_data[0]
    
    def test_len_cache_mode(self, temp_dir, create_test_files):
        """Test dataset length in cache mode."""
        dataset = UnifiedDataset(
            base_path=create_test_files['cache_dir'],
            metadata_path=None,
            repeat=3
        )
        assert len(dataset) == 3 * 3  # 3 cache files * 3 repeats
    
    def test_len_metadata_mode(self, temp_dir, create_test_files, sample_json_data):
        """Test dataset length in metadata mode."""
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=create_test_files['json'],
            repeat=2
        )
        assert len(dataset) == len(sample_json_data) * 2  # 3 items * 2 repeats
    
    def test_check_data_equal(self):
        """Test data equality checking."""
        dataset = UnifiedDataset()
        
        # Test equal data
        data1 = {"a": 1, "b": 2}
        data2 = {"a": 1, "b": 2}
        assert dataset.check_data_equal(data1, data2) is True
        
        # Test unequal data
        data3 = {"a": 1, "b": 3}
        assert dataset.check_data_equal(data1, data3) is False
        
        # Test different lengths
        data4 = {"a": 1}
        assert dataset.check_data_equal(data1, data4) is False
    
    def test_default_image_operator(self):
        """Test default image operator creation."""
        operator = UnifiedDataset.default_image_operator(
            base_path="/test",
            max_pixels=1000*800,
            height=256,
            width=256
        )
        assert isinstance(operator, RouteByType)
        assert len(operator.operator_map) == 2  # str and list types
    
    def test_default_video_operator(self):
        """Test default video operator creation."""
        operator = UnifiedDataset.default_video_operator(
            base_path="/test",
            max_pixels=1000*800,
            height=256,
            width=256,
            num_frames=33
        )
        assert isinstance(operator, RouteByType)
        assert len(operator.operator_map) == 1  # str type only
    
    def test_search_for_cached_data_files(self, temp_dir):
        """Test searching for cached data files recursively."""
        # Create nested directory structure
        nested_dir = os.path.join(temp_dir, "level1", "level2")
        os.makedirs(nested_dir)
        
        # Create cache files at different levels
        cache_files = [
            os.path.join(temp_dir, "cache1.pth"),
            os.path.join(temp_dir, "level1", "cache2.pth"),
            os.path.join(nested_dir, "cache3.pth"),
            os.path.join(temp_dir, "not_cache.txt")  # Should be ignored
        ]
        
        for cache_file in cache_files:
            if cache_file.endswith('.pth'):
                torch.save({"data": "test"}, cache_file)
        
        dataset = UnifiedDataset(base_path=temp_dir)
        dataset.search_for_cached_data_files(temp_dir)
        
        assert len(dataset.cached_data) == 3  # Should find only .pth files
        assert all(path.endswith('.pth') for path in dataset.cached_data)


class TestOperators:
    """Test suite for data processing operators."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL Image."""
        return Image.new('RGB', (200, 100), color='blue')
    
    def test_load_image(self, temp_dir):
        """Test LoadImage operator."""
        # Create test image
        img_path = os.path.join(temp_dir, "test.jpg")
        Image.new('RGB', (50, 50), color='red').save(img_path)
        
        loader = LoadImage()
        image = loader(img_path)
        
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (50, 50)
    
    def test_load_image_no_rgb_convert(self, temp_dir):
        """Test LoadImage operator without RGB conversion."""
        img_path = os.path.join(temp_dir, "test.png")
        Image.new('RGBA', (50, 50), color=(255, 0, 0, 128)).save(img_path)
        
        loader = LoadImage(convert_RGB=False)
        image = loader(img_path)
        
        assert image.mode == "RGBA"
    
    def test_image_crop_and_resize_fixed_size(self, sample_image):
        """Test ImageCropAndResize with fixed dimensions."""
        processor = ImageCropAndResize(height=64, width=64)
        processed = processor(sample_image)
        
        assert processed.size == (64, 64)
    
    def test_image_crop_and_resize_max_pixels(self, sample_image):
        """Test ImageCropAndResize with max_pixels constraint."""
        processor = ImageCropAndResize(max_pixels=5000, height_division_factor=8, width_division_factor=8)
        processed = processor(sample_image)
        
        width, height = processed.size
        assert width * height <= 5000
        assert width % 8 == 0
        assert height % 8 == 0
    
    def test_image_crop_and_resize_division_factors(self, sample_image):
        """Test ImageCropAndResize with division factors."""
        processor = ImageCropAndResize(
            height=None, width=None, max_pixels=10000,
            height_division_factor=16, width_division_factor=32
        )
        processed = processor(sample_image)
        
        width, height = processed.size
        assert height % 16 == 0
        assert width % 32 == 0
    
    def test_to_absolute_path(self):
        """Test ToAbsolutePath operator."""
        processor = ToAbsolutePath(base_path="/base")
        result = processor("subfolder/file.jpg")
        assert result == "/base/subfolder/file.jpg"
        
        # Test with empty base path
        processor_empty = ToAbsolutePath()
        result_empty = processor_empty("file.jpg")
        assert result_empty == "file.jpg"
    
    def test_route_by_type(self, temp_dir):
        """Test RouteByType operator."""
        # Create test image
        img_path = os.path.join(temp_dir, "test.jpg")
        Image.new('RGB', (50, 50), color='green').save(img_path)
        
        def str_processor(x): return f"processed_str_{x}"
        def list_processor(x): return [f"processed_list_{item}" for item in x]
        
        router = RouteByType([
            (str, str_processor),
            (list, list_processor)
        ])
        
        # Test string input
        str_result = router(img_path)
        assert str_result == "processed_str_" + img_path
        
        # Test list input
        list_result = router(["item1", "item2"])
        assert list_result == ["processed_list_item1", "processed_list_item2"]
    
    def test_route_by_type_unsupported_type(self):
        """Test RouteByType with unsupported type."""
        router = RouteByType([(str, lambda x: x)])
        with pytest.raises(ValueError, match="Unsupported data"):
            router(123)  # Integer not supported
    
    def test_route_by_extension_name(self):
        """Test RouteByExtensionName operator."""
        def image_processor(x): return f"image_{x}"
        def video_processor(x): return f"video_{x}"
        
        router = RouteByExtensionName([
            (("jpg", "jpeg", "png"), image_processor),
            (("mp4", "avi"), video_processor)
        ])
        
        # Test image extensions
        assert router("test.jpg") == "image_test.jpg"
        assert router("test.png") == "image_test.png"
        
        # Test video extensions
        assert router("test.mp4") == "video_test.mp4"
        assert router("test.avi") == "video_test.avi"
        
        # Test unsupported extension
        with pytest.raises(ValueError, match="Unsupported file"):
            router("test.txt")
    
    def test_load_torch_pickle(self, temp_dir):
        """Test LoadTorchPickle operator."""
        # Create test pickle file
        test_data = {"tensor": torch.randn(3, 3), "text": "hello"}
        cache_path = os.path.join(temp_dir, "test.pth")
        torch.save(test_data, cache_path)
        
        loader = LoadTorchPickle()
        loaded_data = loader(cache_path)
        
        assert isinstance(loaded_data, dict)
        assert "tensor" in loaded_data
        assert "text" in loaded_data
        assert torch.equal(loaded_data["tensor"], test_data["tensor"])
        assert loaded_data["text"] == test_data["text"]
    
    def test_load_torch_pickle_with_map_location(self, temp_dir):
        """Test LoadTorchPickle with custom map_location."""
        test_data = {"tensor": torch.randn(3, 3)}
        cache_path = os.path.join(temp_dir, "test.pth")
        torch.save(test_data, cache_path)
        
        loader = LoadTorchPickle(map_location="cpu")
        loaded_data = loader(cache_path)
        
        assert loaded_data["tensor"].device.type == "cpu"
    
    def test_to_list(self):
        """Test ToList operator."""
        converter = ToList()
        result = converter("test_item")
        assert result == ["test_item"]
        
        result2 = converter(123)
        assert result2 == [123]
    
    def test_sequential_process(self):
        """Test SequencialProcess operator."""
        def processor(x): return x.upper()
        sequential = SequencialProcess(processor)
        
        result = sequential(["a", "b", "c"])
        assert result == ["A", "B", "C"]
    
    @patch('imageio.get_reader')
    def test_load_video(self, mock_get_reader):
        """Test LoadVideo operator."""
        # Mock video reader
        mock_reader = Mock()
        mock_reader.count_frames.return_value = 100
        mock_reader.get_data.side_effect = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)
        ]
        mock_reader.close = Mock()
        mock_get_reader.return_value = mock_reader
        
        def mock_frame_processor(img):
            return img.resize((32, 32))
        
        loader = LoadVideo(
            num_frames=10,
            time_division_factor=4,
            time_division_remainder=1,
            frame_processor=mock_frame_processor
        )
        
        frames = loader("test_video.mp4")
        
        assert len(frames) == 10
        assert all(isinstance(frame, Image.Image) for frame in frames)
        assert all(frame.size == (32, 32) for frame in frames)
        mock_reader.close.assert_called_once()
    
    @patch('imageio.v3.imread')
    def test_load_gif(self, mock_imread):
        """Test LoadGIF operator."""
        # Mock GIF data
        mock_frames = np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8)
        mock_imread.return_value = mock_frames
        
        def mock_frame_processor(img):
            return img.resize((32, 32))
        
        loader = LoadGIF(
            num_frames=15,
            time_division_factor=4,
            time_division_remainder=1,
            frame_processor=mock_frame_processor
        )
        
        frames = loader("test.gif")
        
        assert len(frames) == 15
        assert all(isinstance(frame, Image.Image) for frame in frames)
        mock_imread.assert_called_once_with("test.gif", mode="RGB")
    
    def test_data_processing_pipeline(self):
        """Test DataProcessingPipeline."""
        def add_one(x): return x + 1
        def multiply_two(x): return x * 2
        
        pipeline = DataProcessingPipeline([add_one, multiply_two])
        result = pipeline(5)
        assert result == 12  # (5 + 1) * 2
    
    def test_data_processing_pipeline_rshift(self):
        """Test DataProcessingPipeline right shift operator."""
        def add_one(x): return x + 1
        def multiply_two(x): return x * 2
        
        op1 = DataProcessingOperatorRaw()
        op1.__call__ = add_one
        op2 = DataProcessingOperatorRaw()
        op2.__call__ = multiply_two
        
        pipeline = op1 >> op2
        result = pipeline(5)
        assert result == 12  # (5 + 1) * 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_empty_metadata_files(self, temp_dir):
        """Test with empty metadata files."""
        # Create empty JSON file
        empty_json = os.path.join(temp_dir, "empty.json")
        with open(empty_json, 'w') as f:
            json.dump([], f)
        
        dataset = UnifiedDataset(metadata_path=empty_json)
        assert len(dataset.data) == 0
        assert len(dataset) == 0
    
    def test_invalid_metadata_extension(self, temp_dir):
        """Test with invalid metadata file extension."""
        invalid_file = os.path.join(temp_dir, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("some text")
        
        # Should raise an error when trying to read as CSV
        with pytest.raises(Exception):
            UnifiedDataset(metadata_path=invalid_file)
    
    def test_missing_image_file(self, temp_dir):
        """Test accessing missing image file."""
        # Create metadata with non-existent image
        metadata = [{"id": 1, "image": "missing.jpg"}]
        json_path = os.path.join(temp_dir, "metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f)
        
        def mock_operator(x):
            # This should fail when trying to load missing file
            if not os.path.exists(x):
                raise FileNotFoundError(f"File not found: {x}")
            return x
        
        dataset = UnifiedDataset(
            base_path=temp_dir,
            metadata_path=json_path,
            data_file_keys=("image",),
            main_data_operator=mock_operator
        )
        
        with pytest.raises(FileNotFoundError):
            dataset[0]
    
    def test_zero_repeat(self):
        """Test dataset with zero repeat."""
        dataset = UnifiedDataset(repeat=0, metadata_path=None)
        dataset.data = [{"id": 1}]
        assert len(dataset) == 0
    
    def test_negative_index_access(self, temp_dir):
        """Test accessing dataset with negative index."""
        metadata = [{"id": 1}, {"id": 2}]
        json_path = os.path.join(temp_dir, "metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f)
        
        dataset = UnifiedDataset(metadata_path=json_path)
        
        # Should handle negative index like regular list
        item = dataset[-1]
        assert item["id"] == 2
    
    def test_very_large_repeat(self):
        """Test dataset with very large repeat value."""
        dataset = UnifiedDataset(repeat=1000000, metadata_path=None)
        dataset.data = [{"id": 1}]
        assert len(dataset) == 1000000
    
    def test_corrupted_cache_file(self, temp_dir):
        """Test with corrupted cache file."""
        corrupted_file = os.path.join(temp_dir, "corrupted.pth")
        with open(corrupted_file, 'w') as f:
            f.write("not a torch file")
        
        dataset = UnifiedDataset(base_path=temp_dir, metadata_path=None)
        dataset.search_for_cached_data_files(temp_dir)
        
        # Should include corrupted file but fail when loading
        assert len(dataset.cached_data) == 1
        
        with pytest.raises(Exception):
            dataset[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])