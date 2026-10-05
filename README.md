# N.O.R.M.A.L.

### Neural Optimized Reasoning Machine for Advanced Learning

**N.O.R.M.A.L.** is an AI engine developed by **Innoxation** for reasoning, natural-language processing, generative capabilities, application development, and ecosystem-aware intelligence.

N.O.R.M.A.L. is being developed across **seven phases**. This repository contains **Phase 1**, the first public development phase.

> **Phase 1 focuses on what N.O.R.M.A.L. should be capable of doing.**

It is intentionally not a complete representation of the final system.

---

## What N.O.R.M.A.L. Can Do

Phase 1 contains implementations and experiments covering several areas of AI:

* 🧠 Multi-level reasoning
* 💬 Natural-language processing
* 🎨 Image generation
* 🎬 Video generation
* 💻 Application generation
* 🌐 Website generation
* 📚 Ecosystem-aware memory
* 🧩 Adaptive response processing
* ⚖️ Legality and policy-oriented reasoning tests
* 🔍 Context and response evaluation
* 🧠 LoRA-based model adaptation

N.O.R.M.A.L. is designed as a foundational intelligence layer for the Innoxation ecosystem rather than as a single-purpose chatbot.

---

# Reasoning System

One of the core ideas behind N.O.R.M.A.L. is that **not every request requires the same amount of reasoning**.

The engine can select different reasoning levels depending on the complexity and ambiguity of the input.

## Level 1 — Low

For simple inputs where little reasoning is required.

```text
Input
  ↓
Thinking
  ↓
Answer
```

Example:

> "Hello Telvin, how are you?"

A simple interaction does not require extensive reasoning.

---

## Level 2 — Mid

Used when additional processing or clarification may be necessary.

```text
Input
  ↓
Working
  ↓
Thinking
  ↓
Adjusting
  ↓
Answer
```

Example:

> "Create Tree."

The request is ambiguous.

Rather than confidently inventing what the user means, the system can recognize that additional context may be necessary and respond accordingly.

For example, it may ask the user to elaborate or continue the conversation to establish intent.

---

## Level 3 — High

Used for complex requests requiring multiple reasoning and evaluation steps.

```text
Input
  ↓
Working
  ↓
Reasoning
  ↓
Assessing
  ↓
Thinking
  ↓
Adjusting
  ↓
Recalculating
  ↓
Refining
  ↓
Answer
```

Example:

> "Explain quantum physics to me like I'm 10."

The system needs to reason about the subject, evaluate the requested explanation level, adjust the response, and refine the result before producing the final answer.

---

## Level 4 — Omni

**Omni** is an advanced reasoning and generation level designed for more complex ecosystem tasks.

It is currently being developed with systems such as **Appgrade**, Innoxation's application-building platform, in mind.

Omni is not considered complete in Phase 1.

---

# Continuous Reasoning Adjustment

N.O.R.M.A.L.'s reasoning does not necessarily end when an intermediate response is produced.

The engine can continue evaluating previous reasoning and adjust its interpretation or response as additional context becomes available.

Conceptually:

```text
Input
  ↓
Reason
  ↓
Response
  ↓
Evaluate previous reasoning
  ↓
Adjust
  ↓
Continue
```

This allows the system to reconsider previous conclusions rather than treating every generated response as permanently fixed.

---

# Ecosystem Memory

N.O.R.M.A.L. uses knowledge of the Innoxation ecosystem to support what is referred to internally as **Ecosystem Memory**.

Ecosystem Memory allows intelligence to operate with awareness of the applications, systems, concepts, and relationships that make up the broader Innoxation environment.

This is particularly important for **Telvin**, where the same intelligence can interact with multiple applications and devices rather than behaving as completely isolated instances.

The objective is continuity of intelligence across the ecosystem.

---

# Model Adaptation

N.O.R.M.A.L. uses **LoRA (Low-Rank Adaptation)** as part of its model adaptation approach.

The adaptation has been tuned specifically around the requirements and behavior expected within the Innoxation ecosystem.

The model layer is only one part of N.O.R.M.A.L.; additional processing and infrastructure exist around it.

---

# Technology

Phase 1 currently uses technologies including:

* **Python**
* **PyTorch**
* **FastAPI**
* **LoRA**
* **Supabase**

Supabase provides part of the accessible infrastructure/surface layer, while additional components operate underneath the exposed architecture.

---

# Integration

Applications within the Innoxation ecosystem can use N.O.R.M.A.L. as an external intelligence service.

For example:

```text
Application
     │
     ▼
N.O.R.M.A.L.
     │
     ├── Reasoning
     ├── Language Processing
     ├── Generation
     ├── Memory
     └── Other Intelligence Services
```

Applications such as **Telvin** currently access N.O.R.M.A.L. through a hosted deployment.

The current development environment uses **Together** for hosting/model access.

---

# Generation

Phase 1 includes experimentation and implementation related to:

### Image Generation

N.O.R.M.A.L. can process requests involving image generation.

### Video Generation

The engine also contains video-generation capabilities.

### Application & Website Generation

N.O.R.M.A.L. contains systems for generating application and website code, forming part of the foundation for future integration with Appgrade.

These capabilities are part of the Phase 1 development surface and should not be interpreted as representing the final capabilities of the seven-phase system.

---

# Testing

Phase 1 has been tested across several areas, including:

* Response generation
* Natural-language processing
* Reasoning behavior
* Different reasoning complexity levels
* Legal and policy-oriented scenarios
* Appropriation-related scenarios
* Fictional and hypothetical situations
* Context interpretation
* Response consistency

One area of ongoing experimentation involves situations where a user provides information that conflicts with established knowledge or appears scientifically impossible.

For example:

> "Scientists have figured out how to give humans superpowers."

The system needs to distinguish between accepting the user's premise as a fictional scenario and treating it as an established factual claim.

This remains an area of active refinement.

---

# Current Limitations

N.O.R.M.A.L. is **not a frontier AI system**.

Phase 1 represents an early development phase of a larger seven-phase project.

Current limitations may include:

* Inconsistent reasoning in certain edge cases
* Conflicts between internal reasoning and final responses
* Incomplete advanced reasoning behavior
* Incomplete Omni capabilities
* Limited robustness across unusual inputs
* Ongoing refinement of factual and fictional-context handling

These limitations are expected at this stage of development.

---

# Development Roadmap

N.O.R.M.A.L. is planned across seven development phases.

```text
Phase 1  →  Public foundational implementation
Phase 2  →  ...
Phase 3  →  ...
Phase 4  →  ...
Phase 5  →  ...
Phase 6  →  ...
Phase 7  →  ...
```

Only Phase 1 is currently being released publicly.

Additional phases may contain systems, data, optimizations, or architectural developments that are intentionally not included in this release.

---

# Innoxation

N.O.R.M.A.L. is developed as part of **Innoxation**.

Innoxation's mission is:

> **Push the limitations of what exists or create something beyond what exists.**

N.O.R.M.A.L. represents the intelligence layer behind part of that vision.

---

## Status

**Current Release:** Phase 1
**Total Planned Phases:** 7
**Status:** Active Development

N.O.R.M.A.L. is an evolving research and engineering project. The public Phase 1 release represents a snapshot of its development rather than its final form.
