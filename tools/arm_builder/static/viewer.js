import { inferFileExtension } from "./api.js";

const DEFAULT_MAP_STYLE = {
  color: "#6ad0ff",
  opacity: 0.3,
  isolevel: 4,
};

const TANGENT_ARROW_LENGTH = 20;
const TANGENT_ARROW_HEAD_LENGTH = 4.8;
const SAVED_TANGENT_ARROW_COLOR = "#ff4fd8";
const SAVED_TANGENT_ARROW_SHAFT_RADIUS = 0.32;
const SAVED_TANGENT_ARROW_HEAD_RADIUS = 0.72;
const DRAFT_TANGENT_ARROW_COLOR = "#ffffff";
const DRAFT_TANGENT_ARROW_SHAFT_RADIUS = 0.28;
const DRAFT_TANGENT_ARROW_HEAD_RADIUS = 0.62;
const SAVED_CONNECTOR_DASH_COLOR = "#ff9e44";
const SAVED_CONNECTOR_DASH_RADIUS = 0.09;
const DRAFT_CONNECTOR_DASH_COLOR = "#ffffff";
const DRAFT_CONNECTOR_DASH_RADIUS = 0.08;
const CONNECTOR_DASH_LENGTH = 1.6;
const CONNECTOR_GAP_LENGTH = 1.0;

function nglVector(point) {
  return new window.NGL.Vector3(point[0], point[1], point[2]);
}

function colorArray(hex) {
  const clean = hex.replace("#", "");
  const value = Number.parseInt(clean, 16);
  return [
    ((value >> 16) & 255) / 255,
    ((value >> 8) & 255) / 255,
    (value & 255) / 255,
  ];
}

function sameScale(scale) {
  return scale.every((value) => Math.abs(value - 1) < 1e-6);
}

function pointsMatch(point1, point2) {
  return point1.every((value, index) => Math.abs(value - point2[index]) < 1e-6);
}

function vectorBetween(fromPoint, toPoint) {
  return toPoint.map((value, index) => value - fromPoint[index]);
}

function vectorLength(vector) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

function scaleVector(vector, scale) {
  return vector.map((value) => value * scale);
}

function addVector(point, vector) {
  return point.map((value, index) => value + vector[index]);
}

function createCoordinateIndex() {
  return {
    residueCentroids: new Map(),
    atoms: new Map(),
    detectedChains: new Set(),
    chainAtomIndices: new Map(),
    residueAtomIndices: new Map(),
  };
}

function averagePoints(points) {
  if (!Array.isArray(points) || points.length === 0) {
    return null;
  }
  const sum = points.reduce(
    (totals, point) => totals.map((value, index) => value + point[index]),
    [0, 0, 0],
  );
  return sum.map((value) => value / points.length);
}

function unitVector(vector) {
  const magnitude = vectorLength(vector);
  if (magnitude <= 1e-6) {
    return null;
  }
  return scaleVector(vector, 1 / magnitude);
}

function selectionKey(...parts) {
  return parts.join("\u0001");
}

function makePointStats() {
  return {
    sum: [0, 0, 0],
    count: 0,
  };
}

function accumulatePointStats(map, key, point) {
  const stats = map.get(key) ?? makePointStats();
  stats.sum[0] += point[0];
  stats.sum[1] += point[1];
  stats.sum[2] += point[2];
  stats.count += 1;
  map.set(key, stats);
}

function accumulateAtomIndex(map, key, atomIndex) {
  const indices = map.get(key) ?? [];
  indices.push(atomIndex);
  map.set(key, indices);
}

function averagePointStats(stats) {
  if (!stats || stats.count === 0) {
    return null;
  }
  return stats.sum.map((value) => value / stats.count);
}

function uniqueSortedNumbers(values) {
  return [...new Set(values)].sort((left, right) => left - right);
}

function finalizeCoordinateIndex(index) {
  return {
    residueCentroids: index.residueCentroids,
    atoms: index.atoms,
    detectedChains: [...index.detectedChains].filter(Boolean).sort(),
    chainAtomIndices: index.chainAtomIndices,
    residueAtomIndices: index.residueAtomIndices,
  };
}

function formatResidueSelection(selection) {
  return `chain ${selection.chain} residue ${selection.residue}`;
}

function formatAtomSelection(selection) {
  return `chain ${selection.chain} residue ${selection.residue} atom ${selection.atom}`;
}

function parsePdbExtendedChainId(line) {
  const prefix = line.substring(20, 21).trim();
  const standard = line.substring(21, 22).trim();
  if (prefix && standard) {
    return `${prefix}${standard}`;
  }
  return standard || prefix || "";
}

function parsePdbCoordinateIndex(text) {
  const index = createCoordinateIndex();
  const lines = text.split(/\r?\n/);
  let atomIndex = 0;

  lines.forEach((line) => {
    if (!line.startsWith("ATOM  ") && !line.startsWith("HETATM")) {
      return;
    }
    const currentAtomIndex = atomIndex;
    atomIndex += 1;

    const atomName = line.substring(12, 16).trim();
    const chainId = parsePdbExtendedChainId(line);
    const residueNumber = Number.parseInt(line.substring(22, 26).trim(), 10);
    const x = Number.parseFloat(line.substring(30, 38).trim());
    const y = Number.parseFloat(line.substring(38, 46).trim());
    const z = Number.parseFloat(line.substring(46, 54).trim());

    if (!chainId || !Number.isFinite(residueNumber) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return;
    }

    const point = [x, y, z];
    index.detectedChains.add(chainId);
    accumulateAtomIndex(index.chainAtomIndices, chainId, currentAtomIndex);
    accumulateAtomIndex(
      index.residueAtomIndices,
      selectionKey(chainId, residueNumber),
      currentAtomIndex,
    );
    accumulatePointStats(
      index.residueCentroids,
      selectionKey(chainId, residueNumber),
      point,
    );
    accumulatePointStats(
      index.atoms,
      selectionKey(chainId, residueNumber, atomName),
      point,
    );
  });

  return finalizeCoordinateIndex(index);
}

function tangentArrowGeometry(anchor, guide, tangent = "direction_point_to_anchor") {
  if (!anchor || !guide || pointsMatch(anchor, guide)) {
    return null;
  }

  const direction = tangent === "anchor_to_direction_point"
    ? vectorBetween(anchor, guide)
    : vectorBetween(guide, anchor);
  const directionUnit = unitVector(direction);
  if (!directionUnit) {
    return null;
  }
  const tip = addVector(anchor, scaleVector(directionUnit, TANGENT_ARROW_LENGTH));
  const headBase = addVector(
    tip,
    scaleVector(directionUnit, -TANGENT_ARROW_HEAD_LENGTH),
  );

  return {
    shaftStart: [...anchor],
    shaftEnd: headBase,
    headBase,
    tip,
  };
}

function addDashedConnector(shape, startPoint, endPoint, color, radius) {
  if (!startPoint || !endPoint || pointsMatch(startPoint, endPoint)) {
    return;
  }

  const direction = vectorBetween(startPoint, endPoint);
  const directionUnit = unitVector(direction);
  const totalLength = vectorLength(direction);
  if (!directionUnit || totalLength <= 1e-6) {
    return;
  }

  let offset = 0;
  while (offset < totalLength) {
    const dashStart = addVector(startPoint, scaleVector(directionUnit, offset));
    const dashEndOffset = Math.min(offset + CONNECTOR_DASH_LENGTH, totalLength);
    const dashEnd = addVector(startPoint, scaleVector(directionUnit, dashEndOffset));
    shape.addCylinder(
      nglVector(dashStart),
      nglVector(dashEnd),
      colorArray(color),
      radius,
    );
    offset += CONNECTOR_DASH_LENGTH + CONNECTOR_GAP_LENGTH;
  }
}

function addWideTangentArrow(shape, anchor, guide, tangent, color, shaftRadius, headRadius) {
  const geometry = tangentArrowGeometry(anchor, guide, tangent);
  if (!geometry) {
    return;
  }

  shape.addCylinder(
    nglVector(geometry.shaftStart),
    nglVector(geometry.shaftEnd),
    colorArray(color),
    shaftRadius,
  );
  shape.addCone(
    nglVector(geometry.headBase),
    nglVector(geometry.tip),
    colorArray(color),
    headRadius,
  );
}

function toAtomSelection(indices) {
  const normalized = uniqueSortedNumbers(indices);
  return normalized.length > 0 ? `@${normalized.join(",")}` : null;
}

function combineSelections(baseSelection, filterSelection) {
  if (!baseSelection) {
    return filterSelection;
  }
  if (!filterSelection) {
    return baseSelection;
  }
  return `(${baseSelection}) and (${filterSelection})`;
}

export class ArmViewer {
  constructor(containerId, callbacks) {
    this.callbacks = callbacks;
    this.stage = new window.NGL.Stage(containerId, {
      backgroundColor: "#051118",
    });
    this.mapComponent = null;
    this.mapRepresentation = null;
    this.modelComponent = null;
    this.modelCoordinateIndex = null;
    this.armComponents = [];
    this.draftComponent = null;
    this.landmarkComponent = null;
    this.mapCenterOffset = [0, 0, 0];
    this.showMap = true;
    this.showModel = true;
    this.modelRepresentation = "cartoon";
    this.focusPresetReferenceGeometry = false;
    this.canonicalPreset = null;

    window.addEventListener("resize", () => this.stage.handleResize(), false);
    this.stage.signals.clicked.add((pickingProxy) => {
      if (!pickingProxy || !pickingProxy.position) {
        this.callbacks.onStatusChange(
          "Click on the rendered map surface or fitted model to capture a point.",
        );
        return;
      }
      this.callbacks.onPointPicked([
        pickingProxy.position.x,
        pickingProxy.position.y,
        pickingProxy.position.z,
      ]);
    });
  }

  async loadMap(file, metadata, contourLevel = DEFAULT_MAP_STYLE.isolevel) {
    this._removeComponent(this.mapComponent);
    this.mapRepresentation = null;

    this.mapComponent = await this.stage.loadFile(file, {
      ext: inferFileExtension(file.name),
      defaultRepresentation: false,
    });

    if (
      typeof this.mapComponent.setScale === "function" &&
      metadata?.viewer_scale_ratio &&
      !sameScale(metadata.viewer_scale_ratio)
    ) {
      this.mapComponent.setScale(metadata.viewer_scale_ratio);
    }

    this.mapRepresentation = this.mapComponent.addRepresentation("surface", {
      ...DEFAULT_MAP_STYLE,
      isolevel: contourLevel,
    });
    this._recenterSceneFromMap();
    this.setMapVisibility(this.showMap);
    this.stage.autoView();
    this.callbacks.onStatusChange(
      "Map loaded. Load a fitted model to generate canonical arms or to manually pick Anchor and Guide Point.",
    );
  }

  async loadModel(file) {
    this._removeComponent(this.modelComponent);
    this.modelCoordinateIndex = null;
    const modelExtension = inferFileExtension(file.name);

    if (modelExtension === "pdb" && typeof file.text === "function") {
      this.modelCoordinateIndex = parsePdbCoordinateIndex(await file.text());
    }

    this.modelComponent = await this.stage.loadFile(file, {
      ext: modelExtension,
      defaultRepresentation: false,
    });

    if (!this.modelCoordinateIndex) {
      this.modelCoordinateIndex = this._buildModelCoordinateIndexFromStructure();
    }

    this._applyModelRepresentation();

    this._applyCenteringToComponent(this.modelComponent);
    this.setModelVisibility(this.showModel);
    this.stage.autoView();
    this.callbacks.onStatusChange(
      "Fitted model loaded. Select a canonical arm preset, or use capture mode for manual picking.",
    );
  }

  setMapVisibility(visible) {
    this.showMap = visible;
    if (this.mapComponent?.setVisibility) {
      this.mapComponent.setVisibility(visible);
    }
  }

  setMapContourLevel(contourLevel) {
    const numeric = Number(contourLevel);
    if (!Number.isFinite(numeric) || numeric <= 0 || !this.mapRepresentation) {
      return;
    }
    if (typeof this.mapRepresentation.setParameters === "function") {
      this.mapRepresentation.setParameters({ isolevel: numeric });
    }
  }

  setModelVisibility(visible) {
    this.showModel = visible;
    if (this.modelComponent?.setVisibility) {
      this.modelComponent.setVisibility(visible);
    }
  }

  setModelRepresentation(representation) {
    const nextRepresentation = representation || "cartoon";
    if (nextRepresentation === this.modelRepresentation) {
      return;
    }
    this.modelRepresentation = nextRepresentation;
    this._applyModelRepresentation();
  }

  setPresetReferenceGeometryFocus(preset, enabled) {
    const nextPreset = preset ?? null;
    const nextPresetKey = nextPreset?.key ?? null;
    const currentPresetKey = this.canonicalPreset?.key ?? null;
    const nextEnabled = Boolean(enabled);

    if (
      nextPresetKey === currentPresetKey &&
      nextEnabled === this.focusPresetReferenceGeometry
    ) {
      return;
    }

    this.canonicalPreset = nextPreset;
    this.focusPresetReferenceGeometry = nextEnabled;
    this._applyModelRepresentation();
  }

  extractCanonicalArms(preset) {
    if (!preset || !Array.isArray(preset.arms) || preset.arms.length === 0) {
      return [];
    }
    if (!this.modelComponent?.structure) {
      throw new Error("Load a fitted model before generating canonical arms.");
    }

    const coordinateIndex = this.modelCoordinateIndex ?? this._buildModelCoordinateIndexFromStructure();
    const translation = this._getComponentPosition(this.modelComponent);
    const missingSelections = [];
    const generatedArms = [];

    preset.arms.forEach((armDefinition) => {
      const anchor = this._extractCanonicalPoint(
        armDefinition.anchor,
        coordinateIndex,
        translation,
        missingSelections,
        `${armDefinition.name} anchor`,
      );
      const guidePoint = this._extractCanonicalPoint(
        armDefinition.guidePoint,
        coordinateIndex,
        translation,
        missingSelections,
        `${armDefinition.name} guide point`,
      );

      if (anchor && guidePoint) {
        generatedArms.push({
          name: armDefinition.name,
          point1_xyz: anchor,
          point2_xyz: guidePoint,
          tangent: armDefinition.tangent || "direction_point_to_anchor",
        });
      }
    });

    if (missingSelections.length > 0) {
      const detectedChains = coordinateIndex.detectedChains.length > 0
        ? coordinateIndex.detectedChains.join(", ")
        : "(none)";
      throw new Error(
        `Could not generate ${preset.label} arms. Missing model selections: ${missingSelections.join("; ")}. Detected chains: ${detectedChains}`,
      );
    }

    return generatedArms;
  }

  _applyModelRepresentation() {
    if (!this.modelComponent) {
      return;
    }
    if (typeof this.modelComponent.removeAllRepresentations === "function") {
      this.modelComponent.removeAllRepresentations();
    }

    const focusedSelection = this._getFocusedModelSelection();

    if (this.modelRepresentation === "cartoon") {
      this.modelComponent.addRepresentation("cartoon", {
        ...(focusedSelection ? { sele: focusedSelection } : {}),
        colorScheme: "chainname",
        opacity: 0.95,
      });
      return;
    }

    if (this.modelRepresentation === "licorice") {
      this.modelComponent.addRepresentation("licorice", {
        sele: combineSelections(focusedSelection, "not hydrogen"),
        colorScheme: "element",
        radius: 0.2,
        opacity: 0.95,
      });
      return;
    }

    if (this.modelRepresentation === "ball-stick") {
      this.modelComponent.addRepresentation("ball+stick", {
        sele: combineSelections(focusedSelection, "not hydrogen"),
        colorScheme: "element",
        scale: 1.35,
        opacity: 0.95,
      });
      return;
    }

    this.modelComponent.addRepresentation("cartoon", {
      ...(focusedSelection ? { sele: focusedSelection } : {}),
      colorScheme: "chainname",
      opacity: 0.78,
    });
    this.modelComponent.addRepresentation("licorice", {
      sele: combineSelections(
        focusedSelection,
        "(protein or nucleic) and not hydrogen",
      ),
      colorScheme: "element",
      radius: 0.18,
      opacity: 0.95,
    });
    this.modelComponent.addRepresentation("ball+stick", {
      sele: combineSelections(focusedSelection, "hetero and not hydrogen"),
      colorScheme: "element",
      scale: 1.6,
      opacity: 0.9,
    });
  }

  renderArms(arms) {
    this.armComponents.forEach((component) => this._removeComponent(component));
    this.armComponents = [];

    arms.forEach((arm) => {
      const shape = new window.NGL.Shape(arm.name);
      shape.addSphere(nglVector(arm.point1_xyz), colorArray(SAVED_TANGENT_ARROW_COLOR), 0.7);
      shape.addSphere(nglVector(arm.point2_xyz), colorArray("#f3c969"), 0.7);
      addDashedConnector(
        shape,
        arm.point1_xyz,
        arm.point2_xyz,
        SAVED_CONNECTOR_DASH_COLOR,
        SAVED_CONNECTOR_DASH_RADIUS,
      );
      addWideTangentArrow(
        shape,
        arm.point1_xyz,
        arm.point2_xyz,
        arm.tangent,
        SAVED_TANGENT_ARROW_COLOR,
        SAVED_TANGENT_ARROW_SHAFT_RADIUS,
        SAVED_TANGENT_ARROW_HEAD_RADIUS,
      );
      const component = this.stage.addComponentFromObject(shape);
      component.addRepresentation("buffer");
      this.armComponents.push(component);
    });
  }

  renderDraftArm(draftArm) {
    this._removeComponent(this.draftComponent);
    this.draftComponent = null;

    if (!draftArm?.point1_xyz && !draftArm?.point2_xyz) {
      return;
    }

    const shape = new window.NGL.Shape("Draft Arm");
    if (draftArm.point1_xyz) {
      shape.addSphere(nglVector(draftArm.point1_xyz), colorArray(DRAFT_TANGENT_ARROW_COLOR), 0.85);
    }
    if (draftArm.point2_xyz) {
      shape.addSphere(nglVector(draftArm.point2_xyz), colorArray("#f3c969"), 0.85);
    }
    addDashedConnector(
      shape,
      draftArm.point1_xyz,
      draftArm.point2_xyz,
      DRAFT_CONNECTOR_DASH_COLOR,
      DRAFT_CONNECTOR_DASH_RADIUS,
    );
    addWideTangentArrow(
      shape,
      draftArm.point1_xyz,
      draftArm.point2_xyz,
      draftArm.tangent,
      DRAFT_TANGENT_ARROW_COLOR,
      DRAFT_TANGENT_ARROW_SHAFT_RADIUS,
      DRAFT_TANGENT_ARROW_HEAD_RADIUS,
    );

    this.draftComponent = this.stage.addComponentFromObject(shape);
    this.draftComponent.addRepresentation("buffer");
  }

  renderLandmarks(landmarks, visible) {
    this._removeComponent(this.landmarkComponent);
    this.landmarkComponent = null;

    if (!visible || !Array.isArray(landmarks) || landmarks.length === 0) {
      return;
    }

    const concrete = landmarks.filter(
      (landmark) =>
        Array.isArray(landmark.position_xyz) && landmark.position_xyz.length === 3,
    );
    if (concrete.length === 0) {
      return;
    }

    const shape = new window.NGL.Shape("Biological Landmarks");
    concrete.forEach((landmark) => {
      shape.addSphere(
        nglVector(landmark.position_xyz),
        colorArray("#ff7d74"),
        0.6,
      );
    });

    this.landmarkComponent = this.stage.addComponentFromObject(shape);
    this.landmarkComponent.addRepresentation("buffer");
  }

  _recenterSceneFromMap() {
    if (!this.mapComponent) {
      return;
    }

    const center = this._getComponentCenter(this.mapComponent);
    this.mapCenterOffset = center.map((value) => -value);
    this._applyCenteringToComponent(this.mapComponent);
    this._applyCenteringToComponent(this.modelComponent);
  }

  _applyCenteringToComponent(component) {
    if (!component || !Array.isArray(this.mapCenterOffset)) {
      return;
    }
    if (typeof component.setPosition === "function") {
      component.setPosition(this.mapCenterOffset);
    }
  }

  _getComponentCenter(component) {
    if (component && typeof component.getCenter === "function") {
      const center = component.getCenter();
      return [center.x, center.y, center.z];
    }
    return [0, 0, 0];
  }

  _buildModelCoordinateIndexFromStructure() {
    const index = createCoordinateIndex();
    let fallbackAtomIndex = 0;

    this.modelComponent.structure.eachAtom((atom) => {
      const atomIndex = Number.isFinite(atom.index) ? atom.index : fallbackAtomIndex;
      fallbackAtomIndex += 1;
      const point = [atom.x, atom.y, atom.z];
      index.detectedChains.add(atom.chainname);
      accumulateAtomIndex(index.chainAtomIndices, atom.chainname, atomIndex);
      accumulateAtomIndex(
        index.residueAtomIndices,
        selectionKey(atom.chainname, atom.resno),
        atomIndex,
      );
      accumulatePointStats(
        index.residueCentroids,
        selectionKey(atom.chainname, atom.resno),
        point,
      );
      accumulatePointStats(
        index.atoms,
        selectionKey(atom.chainname, atom.resno, atom.atomname),
        point,
      );
    });

    return finalizeCoordinateIndex(index);
  }

  _getFocusedModelSelection() {
    if (!this.focusPresetReferenceGeometry) {
      return null;
    }
    if (!Array.isArray(this.canonicalPreset?.referenceGeometry) || this.canonicalPreset.referenceGeometry.length === 0) {
      return null;
    }

    const coordinateIndex = this.modelCoordinateIndex ?? this._buildModelCoordinateIndexFromStructure();
    const focusedAtomIndices = this._collectReferenceGeometryAtomIndices(
      this.canonicalPreset.referenceGeometry,
      coordinateIndex,
    );

    return focusedAtomIndices.length > 0 ? toAtomSelection(focusedAtomIndices) : null;
  }

  _collectReferenceGeometryAtomIndices(referenceGeometry, coordinateIndex) {
    const collected = [];

    referenceGeometry.forEach((selection) => {
      if (selection?.type === "chain") {
        const chainAtomIndices = coordinateIndex.chainAtomIndices.get(selection.chain) ?? [];
        collected.push(...chainAtomIndices);
        return;
      }

      if (selection?.type === "residue_range") {
        for (let residue = selection.residueStart; residue <= selection.residueEnd; residue += 1) {
          const residueAtomIndices = coordinateIndex.residueAtomIndices.get(
            selectionKey(selection.chain, residue),
          ) ?? [];
          collected.push(...residueAtomIndices);
        }
      }
    });

    return uniqueSortedNumbers(collected);
  }

  _extractCanonicalPoint(spec, coordinateIndex, translation, missingSelections, label) {
    if (!spec?.mode) {
      throw new Error(`Unsupported canonical point definition for ${label}.`);
    }

    if (spec.mode === "residue_centroid") {
      const centroid = averagePointStats(
        coordinateIndex.residueCentroids.get(selectionKey(spec.chain, spec.residue)),
      );
      if (!centroid) {
        missingSelections.push(`${label}: ${formatResidueSelection(spec)}`);
        return null;
      }
      return addVector(centroid, translation);
    }

    if (spec.mode === "atom_average") {
      const points = spec.atoms.map((atomSelection) => {
        const averagedAtom = averagePointStats(
          coordinateIndex.atoms.get(
            selectionKey(atomSelection.chain, atomSelection.residue, atomSelection.atom),
          ),
        );
        if (!averagedAtom) {
          missingSelections.push(`${label}: ${formatAtomSelection(atomSelection)}`);
          return null;
        }
        return averagedAtom;
      }).filter(Boolean);

      if (points.length !== spec.atoms.length) {
        return null;
      }

      const averagedPoint = averagePoints(points);
      return averagedPoint ? addVector(averagedPoint, translation) : null;
    }

    throw new Error(`Unsupported canonical point mode "${spec.mode}" for ${label}.`);
  }

  _getComponentPosition(component) {
    if (component?.position) {
      return [
        component.position.x ?? 0,
        component.position.y ?? 0,
        component.position.z ?? 0,
      ];
    }
    return [0, 0, 0];
  }

  _removeComponent(component) {
    if (!component) {
      return;
    }
    this.stage.removeComponent(component);
  }
}
