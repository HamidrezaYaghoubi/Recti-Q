import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const outputRoot = new URL("../out/", import.meta.url);

test("exports the Recti-Q project page", async () => {
  const html = await readFile(new URL("index.html", outputRoot), "utf8");

  assert.match(html, /Recti-Q/);
  assert.match(html, /Robust 4-bit perception/);
  assert.match(html, /Paper coming soon/i);
  assert.doesNotMatch(html, /recti-q-paper\.pdf/i);
  assert.doesNotMatch(html, /chatgpt\.site/i);
});

test("exports the public project assets without a paper PDF", async () => {
  await Promise.all([
    access(new URL("favicon.png", outputRoot)),
    access(new URL("og.png", outputRoot)),
    access(new URL("assets/recti-q-method.png", outputRoot)),
    access(new URL("assets/imagenet-c-results.png", outputRoot)),
  ]);

  await assert.rejects(access(new URL("recti-q-paper.pdf", outputRoot)));
});
