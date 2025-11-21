### 0. Abstract

- **Motivation:** snnTorch SNN training scales poorly with sequence length because neuron recurrences are rolled out sequentially.
    
- **Idea:** reinterpret SNN neuron dynamics as state-space models and present a small taxonomy of PyTorch-compatible parallelization routes.
    
- **Gen1 (merged):** derive a convolutional SSM form of LIF/Leaky and merge **StateLeaky / LinearLeaky** into snnTorch for parallel training + iterative inference fallback.
    
- **Gen2 (merged):** merged a Gen2 spiking SSM with matrix-state associative memory, parallelized via prefix scans, extending the framework beyond Gen1.
    
- **Results:** Gen1 yields large speedups vs Leaky with manageable memory tradeoffs; **Gen2 scales better than StateLeaky at long seqlens** and wins decisively on **NTM Associative Recall**.

- **Takeaway:** snnTorch now supports a Gen1→Gen2 SSM pathway that enables long-context SNN training and introduces associative-memory sequence models in a spiking setting.

### 1. Intro

**1.1 Motivation**

- SNNs: sparse/event-driven advantages, neuromorphic deployment.
    
- But training is GPU-bound and sequential over timesteps → O(T) wall-clock scaling.
    
- This prevents long-range sequence modeling in snnTorch.
    

**1.2 Key observation**

- Many SNN neuron rules are **linear-through-time except reset/spike**, so they resemble SSMs.
    
- If we remove/relax the temporal nonlinearity during training, we can parallelize.
    

**1.3 Contributions (bullets)**

- **Taxonomy:** four recurrence-parallelization families for SSM/SNNs (iterative, convolution, prefix-scan via sums/products, associative scan).
    
- **Gen1 integration:** derived convolutional form of Leaky/LIF without reset → **StateLeaky**, plus ergonomic **LinearLeaky**. Merged into snnTorch.
    
- **Benchmarks:** show speed/memory scaling vs Leaky; show batch-chunking recovers memory with negligible runtime impact.
    
- **Gen2 extension (~⅓):** implement PyTorch-parallel spiking Gen2 SSM (matrix state / associative memory). Show feasibility + advantages on associative-memory tasks; set roadmap to sparse/event-driven Gen2.
    

**1.4 Paper roadmap**

- “Sec 2 reviews background and introduces taxonomy; Sec 3 derives Gen1; Sec 4 benchmarks and snnTorch integration; Sec 5 Gen2 extension; Sec 6 discussion.”

###  2. Background

**2.1 SNN vs RNN recurrence (brief)**

- Contrast dense hidden mixing vs elementwise decay + spike reset.

**2.2 SSM framing**

- Linear-through-time recurrence + output nonlinearity.
    
- Why parallelizable, why useful for GPU training / long sequences.

{FIGURE: Taylor's RNN vs SSM vs SNN}

**2.3 Four parallelization strategies**  

1. **Iterative rollout** (baseline, neuromorphic inference-friendly).
    
2. **Convolution reduction** (time-invariant parameters; PyTorch conv kernels).
    
3. **Prefix-sum / prefix-product scans** (input-dependent dynamics; PyTorch-friendly).
    
4. **General associative scans** (most general; needs custom kernels, clashes with eager PyTorch).

{FIGURE: Four parallelization strategy table}

**2.4 Mapping to SSM “generations”**

{FIGURE: Adapted version of Taylor/Ridger's SSM generation figure}

- Gen1: time-varying scalar decay (GRU-like) → prefix-friendly.
    
- Gen2: matrix state + associative memory → prefix-friendly.
    
- Gen3: aligned write/forget; associative scan needed (hard in PyTorch).
    
- This paper focuses on Gen1 and Gen2 implementations to capture the concrete value of extending snntorch.

### 3. Gen1

**3.1 Starting recurrence (Leaky/LIF)**

- show membrane update + spike/reset.

**3.2 Removing temporal reset for training --> Unrolling --> Convolutional form**

- explain approximation: reset is nonlinearity through time; remove for linear-time dynamics.

- derive closed-form convolution using constant β decay, zero init membrane.
    
- show resulting kernel form.
    
**3.3 Implementation in snnTorch**

- describe **StateLeaky** API + shape handling.
    
- **LinearLeaky** fuse linear + StateLeaky.

- training uses parallel conv; inference can revert to iterative recurrence (neuromorphic-friendly). (Although we do not have the inference implementation).

## 4.  Gen1 Benchmarks + snnTorch integration

{FIGURE: 6 panel memory / inference-time utilization}

{FIGURE: tinystories performance figures}

**4.1 Experimental setup**

- GPU, shapes, seqlens, batch sizes; Leaky vs StateLeaky.

**4.2 Inference-time wins at the little cost**

- log–log plot: runtime vs sequence length.
    
- show StateLeaky asymptotic win; no small-T penalty.

- show Leaky grows due to saved intermediates; StateLeaky larger but predictable.

- show memory recovery ~1 OOM by chunking batch; minimal time cost.
    
**4.3 Application stress test (TinyStories)**

- frame as sequence-modeling sanity check, not “we trained an LLM.”
    
- plot perplexity vs steps and vs wall-clock; StateLeaky advantage over time.
    
**4.4 Ablation: learnable β**

- fixed vs single learnable vs per-channel learnable β.
    
- interpret as evidence that richer state transitions matter → motivates Gen2.

### 5. Gen2 Spiking SSM (Associative State Space Neuron)

**5.1 Motivation from Gen1 results**

- β-learning ablation implies richer state transitions matter.
    
- Taxonomy says Gen2 is the next PyTorch-feasible generation (prefix-scan friendly).
    
- Position Gen2 as snnTorch neuron extending StateLeaky pathway with richer dynamics, also associative memory.
    

**5.2 Gen2 formulation**

- Present Gen2 recurrence with matrix state (write via outer product, read via query).
    
- Clarify single-stream input → internal k/v/q/α projections.
    
- Show exact spiking choice: spike on **S** (resparsify mid-chain).

- Prefix-sum/product scan strategy in PyTorch (input-dependent α allowed).
    
- Mention avoided variant (larger matrix state / 4D materialization) and why current form scales.
    
- snnTorch API / neuron interface design (brief, merger-oriented).

{FIGURE: computation graph / prefix-scan schematic} (unsure if we should include this?)

**5.3 Neuromorphic angle (current + intended)**

- Where dense chains exist today vs Gen1.
    
- Roadmap hooks: top-k sparsification on k/v + scalar column-decay trick.

### 6. Gen2 Evaluation

**6.1 Systems scaling vs Gen1**

- Inference time vs seqlen: Gen2 trends comparably to StateLeaky at long T.
    
- Memory vs seqlen: qualitative similarity to StateLeaky (resource-matched framing).
    
{FIGURE: Gen2 vs StateLeaky time+memory scaling (same format as Sec 4)}

**6.2 Sequence benchmark (TinyStories)**

- Resource-matched comparison to StateLeaky (not parameter-matched).
    
- Perplexity vs batches + vs wall-clock.
    
- Emphasize: faster convergence, less LR sensitivity; compare perplexity trends; (state that we are doing a fair tuned comparison).

{FIGURE: TinyStories Gen2 vs StateLeaky perplexity plots}

**6.3 Associative memory benchmark (decisive Gen2 win)**

- NTM Associative Recall task definition (single-stream binary vectors + delimiter + query).
    
- Scaling sweeps: #pairs, seqlen/delay, width.
    
- Result: Gen2 >> StateLeaky/Gen1, clear separation.

{FIGURE: Associative Recall accuracy/loss scaling curves}

**[CUT. NOT BEING INCLUDED] 6.4 Minimal ablations tied to narrative**

- Spike placement ablation: spike-on-S vs spike-post-q (perf + neuromorphic potential).
    
- (Optional if quick) small top-k pilot showing perf/sparsity tradeoff.

{FIGURE: ablation summary (small, single panel)}

### 7. Discussion

**7.1 What Gen1 enables in snnTorch**

- Parallel training with conv SSM; iterative inference fallback.
    
- Long-context viability in SNN workflows.

**7.2 What Gen2 adds beyond Gen1**

- Better asymptotic scaling than Gen1 in practice.

- Associative / content-addressable memory in spiking SSMs.

**7.3 Limitations**

- Reset-free linearization during training. Although reference that tinystories time-matched figure shows that nonlinearities outside of time can make up for this lack of nonlinearity through seqlen, so it is a minimal limitation.
    
- Dense internal Gen2 ops currently diminish neuromorphic edge.
    
- Gen3 requires associative scan kernels not yet snnTorch-friendly.

**7.4 Roadmap**

- Sparse spiking Gen2 via top-k + scalar decay accumulator.
    
- Potential spiking Gen3 path via torch.compile/Triton/scan kernels.

### 8. Conclusion

- Recap:
    
    - taxonomy + feasibility map,
        
    - Gen1 merged and accelerates snnTorch broadly,
        
    - Gen2 merged and delivers associative memory + better long-T scaling,
        
    - sets clear path to sparse/event-driven Gen2 and future Gen3.