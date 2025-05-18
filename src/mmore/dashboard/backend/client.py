from typing import Optional

import requests


class DashboardClient:
    """
    Client to interact with the dashboard backend from the workers side
    """

    def __init__(self, url: Optional[str]):
        """
        url: the url of the dashboard backend
        """
        self.url = url

    def init_db(self, total_files: int):
        """
        initialize the dashboard
        :param total_files: the total number of files to process
        """
        if self.url is None:
            print("Init db skipped, no url provided")

        try:
            metadata = {"total_files": total_files}
            response = requests.post(f"{self.url}/init-db", json=metadata)
            response.raise_for_status()
        except Exception as e:
            return {"error": str(e)}

    def report(self, worker_id, finished_file_paths) -> bool:
        """
        new process finished execution of a group of files
        :param worker_id: the worker id
        :param finished_file_paths: the list of file paths that have been processed
        :return True is the answer body is True, False otherwise
        """
        if self.url is None:
            print("Report request skipped, no url provided")
            return False
        assert isinstance(finished_file_paths, list)
        assert isinstance(worker_id, str)
        try:
            metadata = {
                "worker_id": worker_id,
                "finished_file_paths": finished_file_paths,
            }
            response = requests.post(f"{self.url}/reports", json=metadata)
            return response.json()
        except Exception as e:
            print(e)
            return False


if __name__ == "__main__":
    # Test the client locally
    backend_url = "http://localhost:8000"
    print(DashboardClient(backend_url).report("42", ["filex"]))
