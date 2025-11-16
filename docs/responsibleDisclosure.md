### Exploiting Predictable Downscaling Algorithms

This project explores image-scaling based prompt injection attacks on multimodal LLMs. These attacks don't require knowledge of a model's internal preprocessing pipelines. They exploit predictable behaviors from common downsampling algorithms.

### Preprocessing is transparent to attackers, opaque to users.

- Even frontier models downsample or compress images internally. Hidden text or adversarial patterns can survive that transformation and become visible only to the model.
- These attacks generalize across platforms
- Even when preprocessing pipelines differ, adversarial images generated with bilinear downsampling can still trigger instructions after resizing.
- So the vulnerability is structural, not vendor-specific.

### Visibility is the strongest defense.


```This work is strictly academic.
All experiments were conducted with benign payloads, and without attempting to exploit or access real user data or privileged actions.
```