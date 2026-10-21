import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { RecognizerSwitch } from "./RecognizerSwitch";
import type { ExperimentalLandmarkInfo } from "../api";

const LOADED: ExperimentalLandmarkInfo = { loaded: true, model_file: "landmark_L2_s0.joblib", model: "logreg", trained_on: null, error: null };

describe("RecognizerSwitch", () => {
  it("renders nothing when no experimental model is available, so normal deployments never see it", () => {
    const { container, rerender } = render(<RecognizerSwitch info={null} value="resnet" onChange={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<RecognizerSwitch info={{ ...LOADED, loaded: false, error: "not found" }} value="resnet" onChange={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("labels ResNet as the default and the landmark model as experimental, naming its file", () => {
    render(<RecognizerSwitch info={LOADED} value="resnet" onChange={vi.fn()} />);
    expect(screen.getByRole("radio", { name: /ResNet.*default, production/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /Hand landmarks.*experimental.*landmark_L2_s0\.joblib/ })).not.toBeChecked();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("switches to the landmark recognizer and offers a one-click way back to ResNet", () => {
    const onChange = vi.fn();
    const { rerender } = render(<RecognizerSwitch info={LOADED} value="resnet" onChange={onChange} />);
    fireEvent.click(screen.getByRole("radio", { name: /Hand landmarks/ }));
    expect(onChange).toHaveBeenCalledWith("landmarks");

    rerender(<RecognizerSwitch info={LOADED} value="landmarks" onChange={onChange} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/Experimental recognizer active.*not validated/s);
    fireEvent.click(screen.getByRole("button", { name: "Back to ResNet" }));
    expect(onChange).toHaveBeenLastCalledWith("resnet");
  });
});
