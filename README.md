# ReviewGuard: Fake Review Detection on Yelp

A hybrid fake review detection system combining transformer-based text semantics with handcrafted reviewer behavioral features to detect fraudulent reviews on the Yelp platform.

## Overview

ReviewGuard uses a two-branch neural architecture:
- **Text Branch**: Fine-tuned RoBERTa-base model extracting 768-dimensional [CLS] embeddings
- **Behavior Branch**: 6 handcrafted reviewer behavioral features
- **Fusion**: Concatenation → 774-dim vector → 2-layer MLP classifier

## Authors

- Rithwik Reddy Donthi Reddy
- Rohan Mukka
