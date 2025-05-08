50% of the time: Handwritten on the left, Printed on the right
Other 50%: Printed on the left, Handwritten on the right

Height matched using padding.

Padding background color is auto-matched from the left crop's border color, for visual continuity.

-------------------
OUTPUT
Sample 0: [ HANDWRITTEN | PRINTED ]
Sample 1: [ PRINTED    | HANDWRITTEN ]
Sample 2: [ HANDWRITTEN | PRINTED ]
... and so on

--------------
Instead of resizing to a fixed size,

The merged image height is set to the taller of the two crops, and

The shorter one is padded (top and bottom) with background color to match height before horizontally stacking.

This preserves more of the original aspect and resolution of the handwritten and printed text while generating visually balanced synthetic samples.
-----------
