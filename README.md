# IOAI Learning Platform

A desktop learning app that takes students from Python basics to IOAI competition level.

## Requirements

- Node.js 18+ (already installed on this machine)
- Internet connection (for Pyodide Python engine and KaTeX fonts)

## First-time setup

```
cd ioai-platform
npm install
```

Then extract Electron manually (Windows only, one-time):

```
node scripts/install-electron.js
```

## Running the app

```
npm start
```

Or for development (with DevTools):

```
npm run dev
```

## Tracks

| Track | Lessons | Description |
|---|---|---|
| 🐍 Python Basics | 11 | Variables to pandas |
| ∑ Math for AI | 6 | Linear algebra, calculus, probability |
| 📊 Classical ML | 9 | Regression, trees, ensembles |
| 🧠 Neural Networks | 8 | Backprop, PyTorch, CNNs |
| 👁 Computer Vision | 7 | ViT, YOLO, CLIP |
| 💬 NLP | 7 | Transformers, BERT, fine-tuning |
| 🏆 IOAI Competition | 7 | Past problems, strategy, checklist |

## Content

Lessons are Markdown files in `content/<track>/`. Quiz data is in companion `-quiz.json` files.

To add a new lesson:
1. Add an entry to `content/courses.json`
2. Create the `.md` file in the track directory
3. Optionally create a `-quiz.json` file for quizzes/exercises
