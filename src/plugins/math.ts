import katex from "katex";
import type { Element } from "hast";
import type { HastPluginDefinition } from "satteri";

function classes(node: Readonly<Element>) {
	const className = node.properties?.className;
	if (Array.isArray(className)) return className.map(String);
	if (typeof className === "string") return className.split(/\s+/);
	return [];
}

function renderMath(value: string, displayMode: boolean) {
	return {
		type: "raw" as const,
		value: katex.renderToString(value, {
			displayMode,
			output: "htmlAndMathml",
			strict: "warn",
			throwOnError: false,
			trust: false,
		}),
	};
}

export function satteriMathPlugin(): HastPluginDefinition {
	return {
		name: "cactus-math",
		element: [
			{
				filter: ["code"],
				visit(node, context) {
					if (!classes(node).includes("math-inline")) return;
					return renderMath(context.textContent(node), false);
				},
			},
			{
				filter: ["pre"],
				visit(node, context) {
					const code = node.children[0];
					if (code?.type !== "element" || code.tagName !== "code") return;
					if (!classes(code).includes("math-display")) return;
					return renderMath(context.textContent(code), true);
				},
			},
		],
	};
}
