# dash line linker
import numpy as np
import mrcfile
import math
from pathlib import Path

# ========= Configurable Parameters =========
pixel_size = 1           # Å per voxel
cylinder_radius_vox = 4    # radius in voxels
length_list_A = [35, 85, 135, 185, 235, 285, 335,500]  # desired cylinder lengths in Å
# dashed parameters (edit as you like)
on_len  = 8   # number of filled z-slices per block
# off_len = 8   # number of empty z-slices per block; set to 0 for solid cylinder
off_len = 0   # number of empty z-slices per block; set to 0 for solid cylinder
out_dir = Path(__file__).parent / "cylinders_out_solid"
out_dir.mkdir(exist_ok=True)


# ========= Function to build one volume =========
def make_cylinder_volume(box_size_vox, length_vox, radius_vox, on_len=0, off_len=0):
    """
    Create a 3D numpy array with a cylinder along Z-axis, centered in XY.

    If off_len == 0  -> solid cylinder (original behavior).
    If off_len  > 0  -> dashed cylinder: repeats blocks of [on_len (filled) + off_len (empty)].
    """
    vol = np.zeros((box_size_vox, box_size_vox, box_size_vox), dtype=np.float32)
    c = box_size_vox // 2

    # 1. Create a 2D circular mask for the XY plane
    y, x = np.ogrid[:box_size_vox, :box_size_vox]
    mask_2d = (x - c)**2 + (y - c)**2 <= radius_vox**2

    # 2. Determine Z-slice indices
    start_z = (box_size_vox - length_vox) // 2
    end_z = start_z + length_vox
    z_indices = np.arange(start_z, end_z)

    # 3. Apply dashed gating if needed
    if off_len > 0:
        if on_len <= 0:
            raise ValueError("on_len must be > 0 when using dashed pattern.")
        period = on_len + off_len
        valid_z = (z_indices - start_z) % period < on_len
        z_indices = z_indices[valid_z]

    # 4. Fill the volume efficiently
    vol[z_indices, ...] = mask_2d

    return vol

# ========= Main loop =========

for length_A in length_list_A:
    length_vox = int(round(length_A / pixel_size))
    box_size_vox = length_vox + 4

    # pass on_len/off_len; if you want solid, use off_len=0
    vol = make_cylinder_volume(box_size_vox, length_vox, cylinder_radius_vox,
                               on_len=on_len, off_len=off_len)

    output_file = out_dir / f"cylinder_len{int(length_A)}A_box{box_size_vox}.mrc"
    with mrcfile.new(output_file.as_posix(), overwrite=True) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = pixel_size
        mrc.update_header_from_data()
        mrc.header.map = b'MAP '
        mrc.update_header_stats()

    print(f"Saved: {output_file}")
    print(f"  Box size: {box_size_vox} vox (~{box_size_vox * pixel_size:.1f} Å)")
    print(f"  Cylinder length: {length_A} Å (~{length_vox} vox)")
    print(f"  Radius: {cylinder_radius_vox * pixel_size:.1f} Å\n")
