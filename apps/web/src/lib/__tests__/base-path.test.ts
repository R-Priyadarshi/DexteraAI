import { describe, expect, it } from "vitest";

import { asset } from "../base-path";

/**
 * `NEXT_PUBLIC_BASE_PATH` is inlined at build time, so these exercise the
 * default (empty) case plus the shapes `asset` must not mangle. The subpath
 * case is covered by the build check in the deploy workflow, which greps the
 * emitted HTML — a unit test here could only re-assert the constant.
 */
describe("asset", () => {
    it("returns root-relative paths unchanged when no base path is set", () => {
        expect(asset("/onnx/hagrid/gesture.onnx")).toBe("/onnx/hagrid/gesture.onnx");
    });

    it("leaves absolute URLs alone", () => {
        // A bundle served from a CDN must not be rewritten into a local path.
        for (const url of [
            "https://cdn.example.com/gesture.onnx",
            "http://cdn.example.com/gesture.onnx",
            "//cdn.example.com/gesture.onnx",
            "blob:https://example.com/abc",
            "data:application/octet-stream;base64,AAAA",
        ]) {
            expect(asset(url)).toBe(url);
        }
    });
});
