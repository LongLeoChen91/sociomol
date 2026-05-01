export const CANONICAL_MODEL_PRESETS = [
  {
    key: "custom",
    label: "Custom / None",
    description: "Do not auto-generate arms. Use manual picking or editing.",
    arms: [],
    referenceGeometry: [],
  },
  {
    key: "6hcj",
    label: "6HCJ 80S ribosome",
    description:
      "Generate A-site and E-site mRNA arms from chain v3 residue centroids.",
    referenceGeometry: [
      { type: "chain", chain: "q3" },
      { type: "chain", chain: "33" },
      { type: "chain", chain: "v3" },
    ],
    arms: [
      {
        name: "A-site mRNA end",
        anchor: {
          mode: "atom_average",
          atoms: [
            { chain: "v3", residue: 34, atom: "C4'" },
          ],
        },
        guidePoint: {
          mode: "atom_average",
          atoms: [
            { chain: "v3", residue: 41, atom: "C4'" },
          ],
        },
        tangent: "anchor_to_direction_point",
      },
      {
        name: "E-site mRNA end",
        anchor: {
          mode: "atom_average",
          atoms: [
            { chain: "v3", residue: 26, atom: "C4'" },
          ],
        },
        guidePoint: {
          mode: "atom_average",
          atoms: [
            { chain: "v3", residue: 20, atom: "C4'" },
          ],
        },
        tangent: "anchor_to_direction_point",
      },
    ],
  },
  {
    key: "4v5d_ba1",
    label: "4V5D 70S ribosome, Biological Assembly 1",
    description:
      "Generate A-site and E-site mRNA arms from chain AX residue centroids.",
    referenceGeometry: [
      { type: "chain", chain: "AX" },
      { type: "chain", chain: "AY" },
      { type: "chain", chain: "AV" },
      { type: "chain", chain: "AW" },
    ],
    arms: [
      {
        name: "A-site mRNA end",
        anchor: {
          mode: "atom_average",
          atoms: [
            { chain: "AX", residue: 21, atom: "C4'" },
          ],
        },
        guidePoint: {
          mode: "atom_average",
          atoms: [
            { chain: "AX", residue: 22, atom: "C4'" },
          ],
        },
        tangent: "anchor_to_direction_point",
      },
      {
        name: "E-site mRNA end",
        anchor: {
          mode: "atom_average",
          atoms: [
            { chain: "AX", residue: 13, atom: "C4'" },
          ],
        },
        guidePoint: {
          mode: "atom_average",
          atoms: [
            { chain: "AX", residue: 12, atom: "C4'" },
          ],
        },
        tangent: "anchor_to_direction_point",
      },
    ],
  },
  {
    key: "2cv5",
    label: "2CV5 nucleosome",
    description:
      "Generate nucleosome arms from explicit DNA base atom averages across chains I and J.",
    referenceGeometry: [
      { type: "residue_range", chain: "J", residueStart: 147, residueEnd: 156 },
      { type: "residue_range", chain: "I", residueStart: 137, residueEnd: 146 },
      { type: "residue_range", chain: "J", residueStart: 283, residueEnd: 292 },
      { type: "residue_range", chain: "I", residueStart: 1, residueEnd: 10 },
    ],
    arms: [
      {
        name: "Nucleosome arm 1",
        anchor: {
          mode: "atom_average",
          atoms: [
            { chain: "J", residue: 147, atom: "N1" },
            { chain: "I", residue: 146, atom: "N3" },
          ],
        },
        guidePoint: {
          mode: "atom_average",
          atoms: [
            { chain: "J", residue: 156, atom: "N3" },
            { chain: "I", residue: 137, atom: "N1" },
          ],
        },
        tangent: "direction_point_to_anchor",
      },
      {
        name: "Nucleosome arm 2",
        anchor: {
          mode: "atom_average",
          atoms: [
            { chain: "J", residue: 292, atom: "N3" },
            { chain: "I", residue: 1, atom: "N1" },
          ],
        },
        guidePoint: {
          mode: "atom_average",
          atoms: [
            { chain: "J", residue: 283, atom: "N1" },
            { chain: "I", residue: 10, atom: "N3" },
          ],
        },
        tangent: "direction_point_to_anchor",
      },
    ],
  },
];

export function getCanonicalModelPresetByKey(key) {
  return CANONICAL_MODEL_PRESETS.find((preset) => preset.key === key)
    ?? CANONICAL_MODEL_PRESETS[0];
}
