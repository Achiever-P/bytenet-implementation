# Presentation Script -- ByteNet Reproduction (~10 minutes)

Written to be spoken, not read off the slide. Split across the three of
you however makes sense -- by slide, by section, whatever's natural.
Timings are estimates for a normal speaking pace; total runtime lands
around 10 minutes. Trim slide 19 first if you're running long -- it's the
densest one.

---

**[Slide 1 -- Title] (~15s)**
"Hi, we're presenting our reproduction of ByteNet -- a paper from IEEE
Transactions on Multimedia that classifies file fragments using computer
vision instead of the usual sequence-based approach."

**[Slide 2 -- Problem Definition] (~35s)**
"Here's the setup. A fragment is just a fixed chunk of a file -- 512
bytes off a disk sector, say -- with nothing attached telling you what
kind of file it came from. No name, no extension. You're trying to guess
the file type from raw bytes alone. That's already hard because the
fragment might come from the middle of a file, so you can't count on
seeing a header. And most existing methods only ever compare a byte to
the bytes next to it -- they never look inside a byte itself. This
matters in digital forensics, recovering files off damaged drives, and
in network security, inspecting traffic with stripped headers."

**[Slide 3 -- Why Interbyte-Only Methods Fall Short] (~40s)**
"So why does looking inside a byte matter? Take JPEG. It compresses its
data with Huffman codes, and Huffman codes are variable-length -- a
single coded value can start in the middle of one byte and end in the
middle of the next. If your model only ever compares whole bytes to each
other, that pattern is just invisible to it, because it doesn't respect
byte boundaries at all. ByteNet's fix is almost mechanical: bit-shift the
fragment seven times, stack the results, and now the bits inside a byte
are just as visible to the model as the bytes themselves."

**[Slide 4 -- Byte2Image: The Transform] (~35s)**
"That bit-shifting gives you a matrix, but it's only 8 pixels wide --
too thin for a CNN to do anything useful with. So there's a second step:
group n consecutive rows together and lay them out side by side, which
widens the image into something closer to square. Then standard
augmentation -- flipping, random erasing, mixup -- since the number of
fragments you have can't grow."

**[Slide 5 -- Byte2Image in Action] (~30s)**
"And this isn't a diagram we drew for the slide -- this is the actual
output of our code, run on a real PNG fragment from our dataset. You can
see the raw bytes on the left, the bit-shifted matrix next to it, and
then the n-gram step visibly widening the image into something with real
structure."

**[Slide 6 -- Code: Bit-Shift] (~25s)**
"I won't walk through this line by line, but this is the actual function
in our repo -- it's basically a direct read of the paper's equation.
Each column is the fragment shifted one more bit than the column before
it."

**[Slide 7 -- ByteNet Architecture] (~35s)**
"On to the classifier itself. It's two branches. A small byte branch --
just one fully-connected layer over the raw bytes, meant to catch
magic-byte patterns like FFD8 for JPEG. And a much deeper image branch --
four stages of residual blocks running over the Byte2Image output, each
one downsampling before the next. Concatenate what both branches produce,
run it through one more layer with softmax, and that's the whole model."

**[Slide 8 -- Code: Dual-Branch Fusion] (~25s)**
"Here's that fusion step in actual code -- x_sf is the byte branch, x_df
is the image branch after its last pooling layer, and concatenating them
is literally this one line here."

**[Slide 9 -- Two Variants] (~30s)**
"The image branch comes in two flavors. ByteResNet uses classic residual
blocks. ByteFormer swaps those for PoolFormer blocks -- pooling instead
of attention, which makes it a lot cheaper: about a third of the
parameters, and something like fifteen times faster per epoch on CPU. We
trained both, all the way through, so we could actually compare them
instead of just trusting the paper's numbers."

**[Slide 10 -- Building a Real Dataset] (~40s)**
"Now, the paper trains on two benchmark datasets, FFT-75 and VFF-16,
and we couldn't get either of those where we were building this. So
instead of testing on random noise, we wrote our own dataset builder that
scans real files already on disk -- Python source, JSON, HTML, PNG, PDF,
gzip archives, ELF binaries -- and cuts 256-byte windows from random
offsets inside them. That's basically how FFT-75 itself was built, just
from whatever documents happened to be around instead of a curated
corpus. Because the offsets are random, a fragment landing on a file's
header is mostly luck, so the model actually has to learn real structure."

**[Slide 11 -- Dataset Build Console Output] (~20s)**
"This is the real output from running that builder -- 2,240 fragments
total, 320 per class, and each class pulled from anywhere between a few
hundred and a few thousand separate source files."

**[Slide 12 -- Training Setup] (~20s)**
"For training we followed the paper's recipe as closely as one CPU core
lets us -- AdamW, cosine learning rate, mixup, the same loss function."

**[Slide 13 -- Training Log Console Output] (~25s)**
"And here's an actual training run -- you can watch test accuracy climb
from basically random guessing up to 73 percent over sixteen epochs.
That's real terminal output, not a number we typed in afterward."

**[Slide 14 -- Results Summary] (~30s)**
"So here's where everything landed. Random guessing on seven classes
gets you about 14 percent. Our full ByteResNet model clears five times
that, at 73.4 percent. And as you'll see in a couple slides, almost all
of that comes from the image branch, not the byte branch."

**[Slide 15 -- Training Curves] (~25s)**
"Quick note on this chart -- train accuracy looks lower and noisier than
test accuracy, and that's expected, not a problem. It's measured on
mixup-blended batches, which are inherently noisier. ByteFormer, on the
right, needed almost four times the epochs just to get close, and still
didn't fully catch up."

**[Slide 16 -- Confusion Matrix] (~35s)**
"This is probably my favorite result, because the errors actually make
sense. PDF, PNG, and gzip fragments keep getting confused with each
other -- and that's because PDF embeds zlib-compressed streams
internally, and PNG uses zlib for its own compression. At the byte level,
compressed data just looks similar regardless of what container it's in.
The original paper hits the exact same wall between its own Archive and
Published categories."

**[Slide 17 -- Branch Ablation] (~30s)**
"Take the image branch away entirely, and accuracy falls to 30.8 percent
-- barely above guessing. Take the byte branch away instead, and it
actually goes up slightly, to 75.4. That's the same story the paper tells
in its own ablation table: the image branch is doing almost all of the
work."

**[Slide 18 -- N-gram Sweep and Variant Comparison] (~30s)**
"Two more checks. Changing the n-gram order barely moves the needle --
about a point of accuracy across the whole range -- which matches what
the paper says about ByteResNet being fairly insensitive to that setting.
And ByteFormer, even with almost four times the training time, settles
well below ByteResNet."

**[Slide 19 -- Honest Discussion] (~45s)**
"We want to be upfront about where our numbers don't quite match the
paper, instead of just showing you the good parts. First, the image-only
ablation actually edges out our full model, which shouldn't happen -- in
the paper, the full model always wins. Our best guess is the byte branch
just needs a lot more data than 320 fragments per class to pull its
weight. Second, ByteFormer underperforms more here than it does in the
paper, even with extra training time. And third, none of these numbers
are directly comparable to the paper's FFT-75 results anyway, because
we're working with a completely different, much smaller dataset. What
ties all of this together is scale, not a mistake in the implementation
-- every claim we could actually test at this size came out the way the
paper predicts."

**[Slide 20 -- Our Contribution] (~25s)**
"Just to be clear about what's ours -- everything on this slide is code
we wrote from scratch: the Byte2Image transform, both network variants,
the dataset builder, and the training loop with the ablation modes built
in. None of it is adapted from the authors' original repository."

**[Slide 21 -- Thank You] (~15s)**
"That's everything -- the full code, every log, every chart in this talk
came straight out of that repository. Happy to take questions."

---

## Delivery notes
- Slides 6, 8, 11, and 13 (code/console) move fast on screen -- don't
  read the code aloud, just point at what matters.
- If splitting speaking parts three ways, natural breakpoints are after
  slide 9 (method complete) and after slide 14 (main result stated) --
  gives each person a roughly even, self-contained chunk.
- The honest-discussion slide (19) is the one worth rehearsing out loud
  a couple of times -- it's the densest and easiest to trip over.
