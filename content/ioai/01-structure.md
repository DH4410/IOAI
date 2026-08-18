---
title: IOAI — Structure and Rules
track: ioai
order: 1
estimatedTime: 30
difficulty: advanced
---

# IOAI — Structure and Rules

Welcome to the IOAI track. If you have reached this point in the curriculum, you have worked through Python, mathematics, classical machine learning, neural networks, computer vision, and natural language processing. This track is different. It is not about learning new theory — it is about learning how to win. This first lesson covers everything you need to know about the competition itself: what it is, how it works, and what it takes to qualify and excel.

---

## 1. What Is IOAI?

The **International Olympiad in Artificial Intelligence (IOAI)** is the world's premier AI competition for high school students. It was founded in 2024 and its first edition was held in **Bulgaria** in August 2024. Like the International Mathematical Olympiad (IMO) or the International Olympiad in Informatics (IOI), IOAI brings together the brightest young minds from around the world to compete at the frontier of their discipline.

IOAI is specifically designed to test **applied AI competency**: the ability to understand theoretical concepts deeply and translate that understanding into working solutions under time pressure. This is not a trivia contest. It is a test of whether you can think like an AI engineer and a scientist simultaneously.

### Why IOAI Matters

- It is the first olympiad dedicated entirely to AI and machine learning, recognizing that AI is now a distinct discipline rather than a sub-field of computer science or mathematics.
- Gold medals from IOAI are recognized by top universities and research labs worldwide.
- The competition exposes students to real-world AI problems — the kind of problems that researchers and engineers solve every day.
- The network you build at IOAI (coaches, other competitors, judges) is extraordinarily valuable for your future career.
- Unlike most academic competitions, IOAI rewards practical skill alongside theoretical knowledge.

### Founding and Governance

IOAI is governed by an international committee that includes representatives from participating countries, AI researchers from top institutions, and education specialists. The competition is held annually, rotating host countries. The first edition was hosted by Bulgaria in 2024, which has a strong tradition of hosting scientific olympiads.

The founding vision of IOAI was to create a competition that:
1. Tests genuine understanding of AI, not memorization
2. Rewards students who can solve novel problems, not just familiar ones
3. Includes an ethical component, recognizing that AI responsibility is central to the field
4. Is accessible to students from countries at different levels of AI education infrastructure

---

## 2. Who Can Compete?

IOAI is open to **high school students** — typically students who are between 14 and 19 years old and have not yet enrolled in a university or higher education institution at the time of the competition.

### Eligibility Requirements

- **Age**: Must be under 20 years old on the day the competition begins.
- **Student status**: Must be enrolled in a secondary school (high school) or equivalent.
- **Citizenship/residency**: Must compete for the country where you are a citizen or permanent resident (rules vary slightly by country — check your National Olympiad Organization).
- **No prior participation**: In most national qualifiers, students who have already participated in IOAI in a previous year are still eligible unless they have graduated. Check your country's specific rules.

### Who Should Aim for IOAI?

IOAI is not for complete beginners. The competition expects students who:

- Have a solid foundation in Python programming
- Understand calculus, linear algebra, and probability at a pre-university level
- Have studied machine learning concepts (regression, classification, neural networks)
- Can implement algorithms from scratch and using standard libraries
- Are comfortable with data analysis and visualization

That said, students who begin serious preparation 12–18 months before the competition and work through a structured curriculum (like this platform) can absolutely qualify. The competition rewards disciplined preparation more than raw genius.

---

## 3. Team Structure

Unlike IMO where students compete entirely individually, IOAI uses a **team-based national representation** model with both team and individual components.

### Team Composition

Each country sends a team of **4 students** to IOAI. These 4 students are selected through the national qualification process. The team is accompanied by:

- **Team Leader / Head Coach**: An adult (usually a university professor, researcher, or experienced educator) who is responsible for the team's logistics and welfare. The Team Leader is the official representative of the country at the IOAI committee level.
- **Deputy Leader / Deputy Coach**: An additional adult who assists the Team Leader.
- **Observers** (optional): Some countries send observers who are allowed to watch (but not participate in) the competition proceedings.

### Individual vs. Team Scoring

The competition has both dimensions:

- **Individual scores**: Each student is scored individually on both the theoretical and practical exams. Individual medals (gold, silver, bronze) are awarded based on individual scores.
- **National team score**: The sum (or top-N sum, depending on the year's rules) of individual scores constitutes the national ranking.

### Coach Role

Coaches are not allowed to communicate with students during the competition itself. Before the competition, coaches may:
- Work through practice problems with students
- Discuss strategy
- Review previous year materials

During the competition, coaches attend separate committee sessions where they translate and review problem statements, and they may raise objections about problem clarity or ambiguity.

---

## 4. Competition Format

IOAI consists of **two major components**: the Theoretical Examination and the Practical Examination. Both are held during the competition week, along with social events, excursions, and the opening/closing ceremonies.

### The Theoretical Examination

The theoretical exam is a written examination that covers:
- Machine learning theory
- Mathematical foundations of AI
- AI ethics and responsible AI
- Conceptual understanding of algorithms

**Format**: Multiple-choice questions (single correct answer) and short-answer questions requiring numerical computation or short written explanations.

**Duration**: Typically 3 hours.

**Topics covered** (see section 6 for full breakdown):
- Supervised learning (regression, classification, decision trees, SVMs)
- Unsupervised learning (clustering, dimensionality reduction, PCA)
- Neural networks (feedforward networks, backpropagation, activation functions)
- Convolutional neural networks (architecture, convolution operation, pooling)
- Recurrent networks and sequence models
- Transformers and attention mechanisms
- Natural language processing (tokenization, word embeddings, language modeling)
- Evaluation metrics (precision, recall, F1, AUC, RMSE, etc.)
- Optimization (gradient descent variants, learning rate schedules, regularization)
- Probability and statistics (Bayes' theorem, distributions, hypothesis testing)
- AI ethics (bias, fairness, privacy, interpretability)

**What the theoretical exam is NOT**: It is not a purely mathematical proof-based exam like IMO. You will not be asked to prove theorems from scratch. You will be asked to compute values, explain concepts, choose the best approach for a given problem, and analyze scenarios.

### The Practical Examination

The practical exam is a **Kaggle-style data science competition** where all teams compete on the same dataset.

**Format**: Students receive a real dataset, a problem statement, and evaluation metric. They must develop a working solution and submit predictions.

**Duration**: Typically 5–8 hours (varies by edition).

**Environment**: Students use provided computers with access to standard data science libraries (Python, NumPy, Pandas, scikit-learn, PyTorch, TensorFlow). Internet access is restricted — no external data, no downloading models. Students may use pre-approved resources.

**Task types** (have included):
- Binary classification (predict yes/no)
- Multi-class classification (predict one of N categories)
- Regression (predict a continuous value)
- Multi-label classification (predict multiple binary labels)
- Potentially: structured prediction, ranking tasks

**Submission**: Students submit predictions as a CSV file. The evaluation system computes the score immediately and students can see their position on a leaderboard.

---

## 5. Scoring System

Understanding the scoring system precisely is critical. Points not understood are points lost.

### Theoretical Exam Scoring

- **Multiple choice questions**: Each question is worth a fixed number of points (typically 1–3 points). Wrong answers may have a small penalty (negative marking) or may simply receive 0 points — check the specific competition year's rules.
- **Short answer questions**: Partial credit is available. A numerical answer that is close but not exactly right may receive partial credit depending on whether the error is in the method or just arithmetic.
- **Total**: Typically 100 points for the theoretical exam.

**Important**: If negative marking applies, leaving a question blank is sometimes better than guessing randomly. However, if you can eliminate even one option, the expected value of guessing among the remaining options is typically positive. We will discuss this in the Strategy lesson.

### Practical Exam Scoring

- The practical exam is typically scored on the **competition metric** defined in the problem statement.
- There are often multiple submission slots. The final submission (or best submission, depending on rules) determines your score.
- Practical exam scores are often normalized or scaled against the baseline and maximum achievable scores.
- **Total**: Typically 100 points for the practical exam.

### Overall Score and Medals

- **Total score** = Theoretical score + Practical score (200 points maximum in most editions).
- Medal cutoffs are determined after all scores are computed, similar to IMO:
  - **Gold**: Top ~8% of competitors (approximately)
  - **Silver**: Next ~17%
  - **Bronze**: Next ~25%
  - **Honorable Mention**: Sometimes awarded for perfect or near-perfect scores on a single exam component.

Note: These percentages are approximate and are adjusted each year to ensure a reasonable distribution of medals.

---

## 6. Theoretical Exam Topic Breakdown

This is the most important section for theoretical preparation. Here is a detailed breakdown of what each topic area covers.

### Machine Learning Fundamentals (High Priority)

- **Bias-variance tradeoff**: Understanding underfitting and overfitting, the decomposition of test error into bias², variance, and irreducible noise.
- **Regularization**: L1 (Lasso), L2 (Ridge), dropout, early stopping — when to use each and the effect on the learned parameters.
- **Cross-validation**: k-fold, stratified k-fold, leave-one-out, time-series cross-validation — why you use them and how they estimate generalization error.
- **Feature selection and engineering**: Filter, wrapper, and embedded methods; mutual information; feature importance from tree models.
- **Ensemble methods**: Bagging (Random Forest), Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM), Stacking.

### Neural Network Theory (High Priority)

- **Feedforward networks**: Architecture, activation functions (ReLU, sigmoid, tanh, GELU, softmax), the universal approximation theorem.
- **Backpropagation**: The chain rule in computation graphs, the vanishing/exploding gradient problem, solutions (batch normalization, residual connections, careful initialization).
- **Optimization**: SGD, momentum, RMSProp, Adam — the update rules you must know cold.
- **Learning rate**: Warmup, cosine annealing, learning rate schedules.
- **Initialization**: Xavier/Glorot, He initialization — why they work.

### Computer Vision (Medium Priority)

- **Convolution**: The convolution operation, filters, feature maps, receptive field.
- **Pooling**: Max pooling, average pooling, global average pooling.
- **Architectures**: LeNet → VGG → ResNet → EfficientNet progression, residual connections.
- **Data augmentation**: Random crop, flip, color jitter, Cutout, Mixup.
- **Transfer learning**: Fine-tuning a pretrained backbone, frozen vs. unfrozen layers.

### NLP (Medium Priority)

- **Text preprocessing**: Tokenization (word, subword — BPE, WordPiece), vocabulary, padding, masking.
- **Word embeddings**: Word2Vec (CBOW, Skip-gram), GloVe — how they are trained, what similarity means.
- **Sequence models**: RNN, LSTM, GRU — the forget/input/output gate equations for LSTM.
- **Attention and Transformers**: Scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder architecture.
- **BERT and GPT**: Pretraining objectives (masked language modeling, causal language modeling), fine-tuning.

### AI Ethics (Lower weight, but guaranteed to appear)

- **Bias in ML**: Sources of bias (data collection, labeling, historical), types (representation bias, measurement bias, aggregation bias).
- **Fairness definitions**: Demographic parity, equalized odds, individual fairness — and why they cannot all be satisfied simultaneously (fairness impossibility theorem).
- **Interpretability**: LIME, SHAP — conceptual understanding.
- **Privacy**: Differential privacy concept, federated learning concept.
- **Responsible deployment**: When should AI NOT be used? High-stakes domains and the need for human oversight.

---

## 7. The Practical Examination — What to Expect

The practical exam is where preparation and execution meet. Here is what actually happens.

### Timeline of the Practical Exam

1. **Problem reveal** (0:00): The problem statement, dataset, and submission format are distributed. You have a few minutes to read before starting.
2. **EDA phase** (0:00–0:30): Understand the data. Load it, look at its shape, check for missing values, understand the target.
3. **Baseline phase** (0:30–1:30): Build a working pipeline end-to-end. A baseline that scores something is better than perfect code that crashes.
4. **Iteration phase** (1:30–4:00): Improve features, try different models, tune hyperparameters.
5. **Ensemble phase** (4:00–5:30): If time permits, combine your best models.
6. **Final submission** (5:30–end): Choose your best submission carefully. Do not change things right before the deadline without good reason.

### What the Judges Look For in the Practical

- **Correctness**: Does your code run? Does it produce valid output?
- **Score on the metric**: Higher is better. This is the primary ranking criterion.
- **Reproducibility**: Some competitions require code submission along with predictions. Code that cannot reproduce your results is penalized.

### Approved Tools and Libraries

Libraries available at IOAI (based on 2024 setup — verify for your year):
- **Python 3.x** (typically 3.10 or 3.11)
- **NumPy, Pandas, SciPy** — data manipulation and scientific computing
- **scikit-learn** — classical ML, preprocessing, metrics, cross-validation
- **LightGBM, XGBoost, CatBoost** — gradient boosting (extremely important for tabular data)
- **PyTorch** — deep learning
- **Matplotlib, Seaborn** — visualization
- **Jupyter Notebook / JupyterLab** — interactive development environment

**NOT available**:
- Internet access (no downloading models, no external APIs)
- Pre-trained large language model weights (unless explicitly provided)
- Proprietary tools

Always check the specific year's approved library list before the competition.

---

## 8. Competition Timeline

### The Annual Cycle

```
September–November: National interest registration
November–January: National training camps begin
January–March: National qualifier exams (written and/or practical)
March–April: National team selection finalized
April–June: National intensive training camps
July–August: IOAI (international competition)
```

### What Happens During Competition Week

**Day 1**: Opening ceremony, team photos, country introductions.
**Day 2**: Theoretical examination (3 hours).
**Day 3**: Day off / excursion (planned by host country).
**Day 4**: Practical examination (5–8 hours).
**Day 5**: Score processing, problem solution presentations.
**Day 6**: Closing ceremony, medal ceremony.
**Day 7**: Departure.

The exact schedule varies by host country and year. Expect social events and excursions interspersed.

---

## 9. IOAI vs. Other Olympiads

Understanding how IOAI compares to other olympiads helps you understand what makes it distinctive.

### IOAI vs. IMO (International Mathematical Olympiad)

| Dimension | IMO | IOAI |
|---|---|---|
| Founded | 1959 | 2024 |
| Focus | Pure mathematics | Applied AI / ML |
| Format | 6 proof-based problems over 2 days | Theory exam + practical coding exam |
| Individual | Yes | Yes (within team) |
| Practical coding | No | Yes |
| Ethics component | No | Yes |
| Prior programming required | No | Yes |

IMO tests creative mathematical reasoning in a proof-based format. IOAI tests applied AI skills with a practical implementation component. They are complementary, not competing.

### IOAI vs. IOI (International Olympiad in Informatics)

| Dimension | IOI | IOAI |
|---|---|---|
| Founded | 1989 | 2024 |
| Focus | Algorithms and data structures | AI and machine learning |
| Practical format | Competitive programming (algorithmic) | Data science (ML modeling) |
| Math required | Discrete math / combinatorics | Calculus, linear algebra, probability |
| Ethics component | No | Yes |

IOI focuses on classical computer science: sorting, graph algorithms, dynamic programming, data structures. IOAI focuses on ML: gradient descent, neural networks, feature engineering, model selection. Students with IOI training have a strong programming foundation but need significant additional preparation for IOAI's ML content.

### IOAI vs. Kaggle Competitions

| Dimension | Kaggle | IOAI Practical |
|---|---|---|
| Duration | Days to months | 5–8 hours |
| Internet | Full access | Restricted |
| Pre-trained models | Full access | Restricted |
| Resources | Unlimited | Fixed hardware |
| External data | Often allowed | Not allowed |

The IOAI practical is a compressed, resource-constrained version of a Kaggle competition. The core skill is the same — building good ML pipelines — but the constraints reward preparation over discovery.

---

## 10. How to Qualify in Your Country

Qualification paths vary significantly by country. Here is the general pattern:

### Step 1: National Interest Registration

Find your country's IOAI National Coordinator. In most countries, this is the national association for computer science or AI education. Register your interest early — information about national qualification processes is often announced 6–9 months before IOAI.

### Step 2: Online/Written Qualification Round

Most countries hold a preliminary online exam that tests:
- Python programming ability
- Basic ML knowledge
- Mathematical foundations

This is typically taken at home or in a supervised school setting.

### Step 3: National Camp

Top performers from the online round are invited to a national training camp. This is where intensive preparation happens:
- Lectures from university professors
- Practice exams mirroring IOAI format
- Group problem-solving sessions

### Step 4: National Final

A final selection exam (or combination of camp performance + exam) determines the 4 students who represent the country.

### Strategy for Qualification

- **Start early**: The students who qualify are typically those who began structured preparation 12–18 months in advance.
- **Work through past national exams**: If your country has held IOAI qualifiers before, practice those problems.
- **Do Kaggle competitions**: Nothing prepares you for the practical exam like actually doing competitions with real data.
- **Study this curriculum**: The curriculum on this platform was designed specifically for IOAI preparation.

---

## 11. Common Disqualification Reasons

Disqualification is rare but real. Avoid these:

### Academic Integrity Violations

- **Copying**: Sharing code or answers with another participant during the competition is an immediate disqualification.
- **Unauthorized communication**: Using a phone or any unauthorized device to receive help from outside.
- **External data**: Bringing in external datasets not provided by the competition.

### Technical Violations

- **Using non-approved libraries**: If a library is not on the approved list, do not import it.
- **Non-reproducible submissions**: Submitting predictions that your code cannot reproduce (due to missing random seeds, for example).
- **Exceeding submission count**: Each competition has a maximum number of submissions per day. Exceeding this is a disqualification in some competitions.

### Practical Advice

- Set all random seeds at the top of your notebook (`np.random.seed(42)`, `torch.manual_seed(42)`, etc.).
- Test your final code from scratch (restart kernel and run all) before the final submission.
- Read the rules at the start of the competition, not halfway through.
- If you are unsure whether something is allowed, ask a proctor immediately.

---

## 12. What the Judges Look For

Beyond raw scores, the committee and coaches pay attention to:

### In Theory

- **Precision of language**: Do you use technical terms correctly?
- **Correct reasoning**: Even if the final answer is wrong, correct method + arithmetic error typically gets partial credit.
- **Avoiding common confusions**: Examples: confusing precision with recall; confusing L1 loss with L1 regularization; confusing correlation with causation.

### In Practice

- **Code quality**: Is your code readable and well-organized? Panicked, disorganized code is harder to debug when something goes wrong.
- **Methodological soundness**: Did you use proper cross-validation? Did you avoid data leakage?
- **Score**: Ultimately, the metric score is the primary criterion. Everything else is secondary.

---

## 13. Building Your Preparation Plan

Given everything above, here is a structured preparation framework:

### 12-Month Timeline

**Months 1–3**: Foundation
- Complete Python, Math, and Classical ML tracks on this platform
- Do 2–3 beginner Kaggle competitions (use existing solutions to learn, then try independently)
- Study linear algebra and calculus at pre-university level

**Months 4–6**: Intermediate
- Complete Neural Networks, Computer Vision, and NLP tracks
- Do 3–5 intermediate Kaggle competitions independently
- Study past IOAI theoretical questions (if available from your country's qualifier)

**Months 7–9**: Advanced
- Complete this IOAI track
- Simulate full competition days (theory exam under timed conditions, practical exam under timed conditions)
- Work through IOAI 2024 and 2025 problems (covered in later lessons)

**Months 10–12**: Peak Preparation
- Focus on weaknesses identified in practice
- Do timed mock exams weekly
- Attend national training camp if selected
- Rest properly the week before

### Daily Practice Habits

- **Code every day**: Even 30 minutes of coding maintains and builds fluency.
- **Read one paper per week**: Choose accessible papers (distillation of large models, BERT, ResNet) to build intuition.
- **Practice mental arithmetic**: Competition theory questions often require quick numerical computation.
- **Explain concepts aloud**: If you cannot explain something to an imaginary student, you do not understand it well enough.

---

## IOAI Format Knowledge Check

```widget
{
  "type": "concept-sort",
  "title": "Theoretical Exam or Practical Exam?",
  "categories": [
    { "name": "Theoretical exam", "color": "#5B5BD6" },
    { "name": "Practical exam", "color": "#22C55E" },
    { "name": "Both exams", "color": "#F97316" }
  ],
  "items": [
    { "text": "Multiple-choice questions on gradient descent concepts", "category": "Theoretical exam" },
    { "text": "Train a model and submit predictions on a test set", "category": "Practical exam" },
    { "text": "Evaluated on cross-validation score and leaderboard", "category": "Practical exam" },
    { "text": "Short-answer question: derive the chain rule application", "category": "Theoretical exam" },
    { "text": "Apply competition strategy (EDA → baseline → iterate)", "category": "Practical exam" },
    { "text": "Academic integrity rules apply", "category": "Both exams" },
    { "text": "Need to know evaluation metrics deeply", "category": "Both exams" },
    { "text": "Questions on AI ethics and societal impact", "category": "Theoretical exam" }
  ]
}
```

---

## 14. Summary

IOAI is a dual-format competition: a theoretical examination testing ML knowledge, math, and ethics; and a practical examination that is a compressed Kaggle-style data challenge. It was founded in 2024, held in Bulgaria for its first edition, and has grown rapidly.

The keys to success are:
1. **Deep theoretical understanding**, not memorization
2. **Practical coding fluency** — you must be able to build a working ML pipeline under pressure
3. **Metric awareness** — always know what you are optimizing
4. **Time management** — the competition is a race against the clock as much as against other competitors
5. **Academic integrity** — your reputation is more valuable than any medal

The rest of this track will prepare you for everything that follows. Let us get to work.

---

*Next: Lesson 2 — Competition Strategy*
