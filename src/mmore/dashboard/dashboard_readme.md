# m(m)ore Dashboard Documentation

# 1. Overall Structure

Before setting up the dashboard, it is useful to understand how it works. You can think of the dashboard as being made up of 4 separate parts:

![image.png](doc_images/image.png)

| **Frontend:** the actual dashboard user interface (UI), and what will be displayed on your screen. | ![Frontend](doc_images/image%201.png) |
| **Database:** the database which stores information about the file processing. | ![Database](doc_images/image%202.png) |
| **Processing Pipeline:** the pipeline processing your documents for which you want to be able to visualize on the dashboard. | ![Pipeline](doc_images/image%203.png) |
| **Backend Server:** *backend* is what we call the server that acts like the middle man to the 3 elements above. It receives information from the processing pipeline, stores and retrieves data from the database and sends information to be displayed on the frontend dashboard. | ![Backend](doc_images/image%204.png) |


# 2. Setup

Each element shown above is created in a different terminal. This means that you will need to have 4 terminals running to launch the dashboard successfully.

## Terminal 1: MongoDB  Setup

Official documentation for MongoDB setup can be found [here](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) (Ubuntu 22.04 Jammy release). 

![image.png](doc_images/image%205.png)

> **Note:** These steps must be repeated each time you submit a new runai job. 

### Manual Setup Instructions

1. **Install required tools**

```bash
sudo apt-get install gnupg curl
```

- `gnupg`: Encryption tool for secure communication and data storage
- `curl`: Command-line tool for transferring data with URLs

1. **Add MongoDB's GPG Key**

```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor
```

This command:

- Downloads MongoDB's digital signature key
- Converts it to binary format
- Stores it in the system's keyring for package verification

1. **Add MongoDB Repository**

```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
```

This adds the official MongoDB repository to your package sources, specifically for Ubuntu 22.04 (Jammy).

1. **Install MongoDB**

```bash
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo apt-get install -y mongodb-org=8.0.5 mongodb-org-database=8.0.5 mongodb-org-server=8.0.5 mongodb-mongosh mongodb-org-mongos=8.0.5 mongodb-org-tools=8.0.5
```

> **Note**: You will be prompted to select your time zone during installation. If you are in Lausanne, enter '8' for Europe and then '63' for the timezone.

1. **Create Data Directory**

```bash
mkdir -p ~/mongodb
```

Creates a directory in root folder to store MongoDB data files. 

> **Important**: This directory and all data will be deleted when the job terminates as the home directory is not persistent storage. 

1. **Start the MongoDB Server**

```bash
mongod --bind_ip_all --dbpath ~/mongodb
```

This starts MongoDB with the following configuration:

- `--bind_ip_all` : Accepts connections from any IP address
- `--dbpath ~/mongodb` : Specifies where MongoDB should store its data files

In your current terminal you should see MongoDB logs and messages appearing. This means that your terminal is successfully running the MongoDB server in your terminal. 

> **Important**: : Keep this terminal window open. MongoDB runs in the foreground and closing the terminal will shut down the server. The server listens on port 27017 by default.

1. **Shutting down MongoDB**

To stop the MongoDB server, press `Ctrl + C` in the terminal where it's running. This initiates a clean shutdown.

### Automated Setup Script

For convenience, you can save the following bash script in your project directory and make it executable:

```bash
#!/bin/bash
# MongoDB startup script

# Install MongoDB if not already installed
which mongod > /dev/null # put the output into nothing 
if [ $? -ne 0 ]; then #If the exit status of the previous command is not 0
  echo "Installing MongoDB..."
  sudo apt-get update
  sudo apt-get install -y gnupg curl
  curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
    sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
    --dearmor
  echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
  sudo apt-get update
  sudo apt-get install -y mongodb-org
  sudo apt-get install -y mongodb-org=8.0.5 mongodb-org-database=8.0.5 mongodb-org-server=8.0.5 mongodb-mongosh mongodb-org-mongos=8.0.5 mongodb-org-tools=8.0.5
fi

# Create MongoDB data directory if it doesn't exist
mkdir -p ~/mongodb

# Start MongoDB
echo "Starting MongoDB..."
mongod --bind_ip_all --dbpath ~/mongodb
```

To use this script:

1. Save it as `start_mongodb.sh` in your project directory
2. Make it executable: `chmod +x start_mongodb.sh`
3. Run it each time you need MongoDB: `./start_mongodb.sh`

This script automatically checks if MongoDB is installed, installs it if needed, and starts the server.

## Terminal 2: **Backend Setup**

This backend serves as the bridge between the **database,** the **frontend** and **processing pipeline**, providing a clean API to interact with the data without direct database access.

![image.png](doc_images/image%206.png)

### Setup Instructions

1. **Activate Virtual Environment** 

```bash
# If using venv (standard Python virtual environment)
source .venv/bin/activate
```

1. **Install Dependencies**

```bash
pip install -r src/mmore/dashboard/backend/backend_requirements.txt
```

1. **Configure MongoDB Connection**

```bash
export MONGODB_URL="mongodb://localhost:27017"
```

Sets the environment variable to tell the backend how to connect to MongoDB instance

> **Important**: Your MongoDB server should be active before starting the backend.

1. **Start the Backend Server**

Run the backend on port 8000

```bash
python -m uvicorn src.mmore.dashboard.backend.main:app --host 0.0.0.0 --port 8000
```

This command:

- Starts the Uvicorn ASGI server
- Loads the FastAPI application from the main.py file
- Binds it to all network interfaces (0.0.0.0)
- Makes it listen on port 8000

> **Important**: Keep this terminal window open. The backend runs in the foreground and closing the terminal will shut down the server.

1. **Verify the Backend is Running**

You can check if the backend is running correctly by accessing [http://localhost:8000](http://localhost:8000). You should see a response like: `{"message": "Hello World"}`

For API documentation, visit [http://localhost:8000/docs](http://localhost:8000/docs). This will show the automatically all available endpoints.

---

The next step is to set up the frontend that will communicate with this backend to provide a user interface for monitoring and control.

## **Terminal 3: Frontend Setup**

This frontend serves as the user-facing component of the system, providing an  interface for monitoring and controlling the processing pipeline without requiring direct interaction with the database or backend code.

![image.png](doc_images/image%207.png)

1. **Load Node Version Manager**

```bash
source /usr/local/nvm/nvm.sh
```

This loads Node Version Manager (NVM) into your current shell session. NVM is necessary because the frontend requires a specific version of Node.js that differs from the default version installed on the system.

1. **Install and Activate Node.js Version 23** 

```bash
sudo -i # give root privileges 
nvm install 23
exit # exit root 
nvm use 23
```

This sequence:

- Starts a shell with root privileges (necessary for the installation)
- Uses NVM to install Node.js version 23
- Exits the root shell
- Sets version 23 as the active Node.js version for your current session
1. **Install Dependencies**

```bash
cd src/mmore/dashboard/frontend # navigte to frontend directory
npm install
```

This command uses NPM (Node Package Manager) to install all JavaScript dependencies defined in the package.json file. These are libraries and frameworks needed by the frontend.

1. **Configure Backend URL** 

```bash
export VITE_BACKEND_API_URL="http://localhost:8000"
```

Sets an environment variable that tells the frontend where to find the backend API. Vite (the build tool) will use this variable during development.

1. **Start Frontend Server**

```bash
npm run dev
```

Executes the development script defined in package.json, and starts a local development server for the frontend application. The terminal will show the URL where the frontend is available (typically [http://localhost:5173](http://localhost:5173/))

## Terminal 4: Run Process Pipeline

To complete the dashboard setup, you need to run a process module that will generate data for visualization in the UI. 

![image.png](doc_images/image%208.png)

1. **Modify Configuration File**

Update `config.yaml` to match the backend url:`dashboard_backend_url: http://localhost:8000` 

1. **Activate Virtual Environment** 

```bash
# If using venv (standard Python virtual environment)
source .venv/bin/activate
```

1. **Run the Process Module**

```bash
python -m src.mmore.processing.run_processor --config examples/process/config.yaml
```

1. **Monitor the Dashboard**

Once the process module is running, it will:
1. Process files from the input directory specified in the config
2. Send progress reports to the MongoDB database via the backend API
3. Update the dashboard UI in real-time

Return to your browser where the frontend is running to see the visualization of the processing progress.

###