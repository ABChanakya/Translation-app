class ImgComparisonSlider extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: "open" });
    }

    connectedCallback() {
        if (this.shadowRoot.children.length) {
            return;
        }

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    position: relative;
                    overflow: hidden;
                    background: #111;
                    --divider-width: 4px;
                    --divider-color: #ef6c3b;
                    --handle-size: 44px;
                }

                .frame {
                    position: relative;
                    width: 100%;
                    line-height: 0;
                    user-select: none;
                }

                .first,
                .second {
                    display: block;
                    width: 100%;
                    height: auto;
                }

                .second-wrap {
                    position: absolute;
                    inset: 0;
                    overflow: hidden;
                    width: 50%;
                    border-right: var(--divider-width) solid var(--divider-color);
                    box-sizing: border-box;
                }

                .range {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                    opacity: 0;
                    cursor: ew-resize;
                    margin: 0;
                    z-index: 3;
                }

                .handle {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: var(--handle-size);
                    height: var(--handle-size);
                    transform: translate(-50%, -50%);
                    border-radius: 999px;
                    background: rgba(239, 108, 59, 0.95);
                    color: #fff;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    font-weight: 700;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
                    pointer-events: none;
                    z-index: 2;
                }

                .handle::before {
                    content: "↔";
                }

                ::slotted(img) {
                    display: block;
                    width: 100%;
                    height: auto;
                    object-fit: contain;
                }
            </style>
            <div class="frame">
                <slot class="first" name="first"></slot>
                <div class="second-wrap">
                    <slot class="second" name="second"></slot>
                </div>
                <div class="handle" aria-hidden="true"></div>
                <input class="range" type="range" min="0" max="100" value="50" aria-label="Image comparison slider">
            </div>
        `;

        const range = this.shadowRoot.querySelector(".range");
        const secondWrap = this.shadowRoot.querySelector(".second-wrap");
        const handle = this.shadowRoot.querySelector(".handle");

        const update = () => {
            const value = `${range.value}%`;
            secondWrap.style.width = value;
            handle.style.left = value;
        };

        range.addEventListener("input", update);
        update();
    }
}

if (!customElements.get("img-comparison-slider")) {
    customElements.define("img-comparison-slider", ImgComparisonSlider);
}
