import { beforeEach, describe, expect, it } from "vitest";
import { type Landmark } from "../gesture-engine";
import { GestureStore, GESTURE_PACK_FORMAT } from "../gesture-store";

/** A structurally valid 21-point sample. */
const sample = (): Landmark[] =>
  Array.from({ length: 21 }, (_, i) => ({ x: i / 21, y: i / 42, z: 0 }));

/** localStorage stand-in — vitest runs in node, which has no DOM. */
class MemoryStorage {
  private map = new Map<string, string>();
  getItem = (k: string) => this.map.get(k) ?? null;
  setItem = (k: string, v: string) => void this.map.set(k, v);
  removeItem = (k: string) => void this.map.delete(k);
  clear = () => this.map.clear();
  key = () => null;
  length = 0;
}

beforeEach(() => {
  // Minimal stand-ins for the browser globals the store touches; vitest runs
  // in node, which provides neither.
  (globalThis as Record<string, unknown>).window = {};
  (globalThis as Record<string, unknown>).localStorage = new MemoryStorage();
});

const pack = (gestures: unknown[]) => ({
  format: GESTURE_PACK_FORMAT,
  version: 1,
  exportedAt: new Date().toISOString(),
  gestures,
});

describe("GestureStore packs", () => {
  it("round-trips gestures through export and import", () => {
    const a = new GestureStore();
    a.addGesture("salute", [sample(), sample()]);
    const exported = a.exportPack();

    // A second store standing in for another device: same pack, empty storage.
    localStorage.clear();
    const b = new GestureStore();
    const report = b.importPack(exported);

    expect(report.imported).toBe(1);
    expect(b.getGestures()).toHaveLength(1);
    expect(b.getGestures()[0].name).toBe("salute");
    expect(b.getGestures()[0].samples).toHaveLength(2);
  });

  it("assigns fresh ids so a re-import cannot collide", () => {
    const a = new GestureStore();
    a.addGesture("salute", [sample()]);
    const exported = a.exportPack();

    localStorage.clear();
    const b = new GestureStore();
    b.importPack(exported);
    expect(b.getGestures()[0].id).not.toBe(exported.gestures[0].id);
  });

  it("refuses a file that is not a gesture pack", () => {
    const s = new GestureStore();
    expect(() => s.importPack({ hello: "world" })).toThrow(/not a dextera gesture pack/i);
  });

  it("refuses an unsupported pack version", () => {
    const s = new GestureStore();
    expect(() => s.importPack({ ...pack([]), version: 99 })).toThrow(/version/i);
  });

  it("rejects samples with the wrong landmark count", () => {
    // A short sample would produce NaN distances in the k-NN matcher and
    // quietly degrade recognition for every gesture, not just this one.
    const s = new GestureStore();
    const report = s.importPack(
      pack([{ name: "bad", samples: [[{ x: 0, y: 0, z: 0 }]], createdAt: 1 }])
    );
    expect(report.imported).toBe(0);
    expect(report.rejected[0].reason).toMatch(/21 landmarks/);
  });

  it("rejects non-finite coordinates", () => {
    const s = new GestureStore();
    const broken = sample();
    broken[3] = { x: NaN, y: 0, z: 0 };
    const report = s.importPack(pack([{ name: "nan", samples: [broken], createdAt: 1 }]));
    expect(report.imported).toBe(0);
    expect(report.rejected[0].reason).toMatch(/finite/);
  });

  it("rejects a gesture with no samples", () => {
    const s = new GestureStore();
    const report = s.importPack(pack([{ name: "empty", samples: [], createdAt: 1 }]));
    expect(report.rejected[0].reason).toMatch(/no samples/);
  });

  it("skips duplicates rather than overwriting the user's own gesture", () => {
    const s = new GestureStore();
    const mine = s.addGesture("wave", [sample()]);
    const report = s.importPack(pack([{ name: "wave", samples: [sample(), sample()], createdAt: 1 }]));

    expect(report.imported).toBe(0);
    expect(report.skippedDuplicates).toEqual(["wave"]);
    expect(s.getGestures()).toHaveLength(1);
    expect(s.getGestures()[0].id).toBe(mine.id);
    expect(s.getGestures()[0].samples).toHaveLength(1);
  });

  it("imports the good gestures from a partly broken pack", () => {
    const s = new GestureStore();
    const report = s.importPack(
      pack([
        { name: "good", samples: [sample()], createdAt: 1 },
        { name: "bad", samples: [[]], createdAt: 1 },
        { name: "", samples: [sample()], createdAt: 1 },
      ])
    );
    expect(report.imported).toBe(1);
    expect(report.rejected).toHaveLength(2);
    expect(s.getGestures().map((g) => g.name)).toEqual(["good"]);
  });

  it("renames a gesture but refuses a name already in use", () => {
    const s = new GestureStore();
    const a = s.addGesture("one", [sample()]);
    s.addGesture("two", [sample()]);

    expect(s.renameGesture(a.id, "three")).toBe(true);
    expect(s.getGestures()[0].name).toBe("three");
    expect(s.renameGesture(a.id, "two")).toBe(false);
    expect(s.renameGesture(a.id, "  ")).toBe(false);
  });
});
