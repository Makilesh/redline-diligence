import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { MotionGlobalConfig } from "framer-motion";
import { afterEach } from "vitest";

// Animations complete instantly, so a test never waits on an exit transition.
MotionGlobalConfig.skipAnimations = true;

afterEach(() => {
  cleanup();
});
