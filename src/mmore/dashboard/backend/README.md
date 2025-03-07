### Install dependencies
```bash
pip install -r backend_requirements.txt
```
### Run the backend on port 8000
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Port Forwarding
The front end will need to access this backend on port 8000. Note that you may need to do a port forwarding if you are running this on a remote server.
If you use kube, you can use the following command to port forward:
```bash
kubectl port-forward pods/mypod LOCAL:REMOTE
# example : kubectl port-forward pods/job-8af559ef097e-master-0 8000:8000
```
This allows you to access the backend on `http://localhost:8000` on your local machine.
My running the frontend on your local machine you can access the backend on `http://localhost:8000` as well.