# Frontend – Food Classification App

This is a Next.js (TypeScript) client that allows users to upload an image and receive a classification prediction (Pizza, Steak, or Sushi) from a backend ML API.

## Features

* Built with Next.js 14 using the `/app` directory structure.
* Uploads an image to the FastAPI backend.
* Displays the predicted class and confidence scores.
* Shows a live preview of the selected image before submission.
* Uses Tailwind CSS for layout and styling.

## API Dependency

This frontend requires the backend service to be running and accessible at:

```
http://dlc-jupyter:8000/api/v1/upload/image
```

You can adjust the URL in `src/app/page.tsx` if your backend is hosted elsewhere.

## Running the Frontend

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Open your browser at [http://localhost:3000](http://localhost:3000)

## Folder Structure

```
frontend/
├── public/              # Static assets
├── src/                 # Application code
│   └── app/page.tsx     # Main page component
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── README.md
```

## Notes

* The UI assumes class probabilities are returned as a dictionary mapping class names to probabilities.
* All styling is done with Tailwind CSS utility classes.
* No backend logic is included here. This is purely a client-side UI app.
