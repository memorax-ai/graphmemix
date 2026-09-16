# Reader context capacity

`mmmb run-method ... --reader-max-model-len 65536` sets the answer client's
combined prompt and output token limit. The default remains 32768 for every
method. Router, caption, and memory-extraction clients keep their existing limits.

This option does not resize or restart a model service. The selected Reader
endpoint must separately support the requested capacity. Preserve the input;
the client only reduces the output allowance near the limit and still rejects
an input that fills or exceeds the configured window. No evidence is silently
truncated. A larger window does not guarantee every benchmark question fits.

Use a separate predictions directory when changing context capacity, and record
the service configuration and this option with the experiment. Compatible memory
indexes can be reused. Verify previously failing questions before a full run.
