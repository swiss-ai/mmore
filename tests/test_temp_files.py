import unittest
from pathlib import Path
from PIL import Image
from src.mmore.process.utils import _save_temp_image
from multiprocessing import Pool

def create_temp_file(_):
    image = Image.new("RGB", (100, 100), color="red")
    return _save_temp_image(image)

class TestTempFiles(unittest.TestCase):
    def test_temp_file_uniqueness_parallel(self):
        num_processes = 10
        files_per_process = 1000 
        total_files = num_processes * files_per_process

        try:
            with Pool(num_processes) as pool:
                results = pool.map(create_temp_file, range(total_files))
            self.assertEqual(len(results), len(set(results)), "Temporary file names should be unique")

        finally:
            for file_path in results:
                Path(file_path).unlink()

if __name__ == "__main__":
    unittest.main()
