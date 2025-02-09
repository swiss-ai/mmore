# MMORE Dashboard

#### Frontend server made with React + TypeScript + Vite and ESLint

After starting the backend server (see `/src/mmore/dashboard/backend/README.md`) on the cluster, you can connect to it
with this front end user interface.

# Install vite 6.0.3 and node-v23.5.0

```bash
npm install vite@6.0.3
```

### Setup & run

First specify the backend URL in the `.env` file, example:

```text
VITE_BACKEND_API_URL=http://localhost:8000
```

PLace yourself at the `src/mmore/dashboard/frontend` directory:

```bash
Then install the dependencies:
```bash
npm install
```

To run the fronted you need to run the following commands:

```bash
npm run build
npm run preview
```

### Development mode

For development run the following command:

```bash
npm run dev
```