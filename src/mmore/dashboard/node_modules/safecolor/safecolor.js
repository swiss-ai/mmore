/**
 * @author jessuni
 */


const _CONFIG = {
  color: [0, 0, 0],
  contrast: 4.5,
  step: 0.05,
}

class SafeColor {
  /**
   * Create a SafeColor instance with options
   * @param {Array} color 8bit RGB value of a given color in the form of [R, G, B]
   * @param {Number} contrast contrast ratio of foreground color and background color https://www.w3.org/TR/WCAG20/#contrast-ratiodef
   */
  constructor(options = {}) {
    this.color = options.color || _CONFIG.color
    this.contrast = options.contrast || _CONFIG.contrast
    this.step = _CONFIG.step
    this.hash = 0
  }

  _rgb2hsl([r, g, b]) {
    r /= 255
    g /= 255
    b /= 255
    const cmax = Math.max(r, g, b)
    const cmin = Math.min(r, g, b)
    const chroma = cmax - cmin
    let l = (cmax + cmin) / 2
    let h, s
    if (!chroma) {
      h = s = 0
    } else {
      s = chroma / (1 - Math.abs(2 * l - 1))
      if (cmax === r) {
        h = ((g - b) / chroma) % 6
      } else if (cmax === g) {
        h = (b - r) / chroma + 2
      } else {
        h = (r - g) / chroma + 4
      }
      h = Math.round(h * 60)
      h = h < 0 ? h + 360 : h
    }
    return [h, s, l]
  }

  _hsl2rgb([h, s, l]) {
    let c = (1 - Math.abs(2 * l - 1)) * s
    let x = c * (1 - Math.abs((h / 60) % 2 - 1))
    let m = l - c / 2
    let r = 0
    let g = 0
    let b = 0
    if (h >= 0 && h < 60) {
      r = c
      g = x
      b = 0
    } else if (h >= 60 && h < 120) {
      r = x
      g = c
      b = 0
    } else if (h >= 120 && h < 180) {
      r = 0
      g = c
      b = x
    } else if (h >= 180 && h < 240) {
      r = 0
      g = x
      b = c
    } else if (h >= 240 && h < 300) {
      r = x
      g = 0
      b = c
    } else if (h >= 300 && h < 360) {
      r = c
      g = 0
      b = x
    }
    r = Math.round((r + m) * 255)
    g = Math.round((g + m) * 255)
    b = Math.round((b + m) * 255)
    return [r, g, b]
  }

  /**
   * Calculate the luma of a given sRGB array
   * @param {Array} sRGB
   * @return luma of the color
   */
  calLuma(sRGB) {
    const [linearR, linearG, linearB] = sRGB.map(v => {
      const decimal = v / 255
      return decimal <= 0.04045 ? decimal / 12.92 : Math.pow((decimal + 0.055) / 1.055, 2.4)
    })
    return linearR * 0.2126 + linearG * 0.7152 + linearB * 0.0722
  }

  /**
   * Calculate the valid luma range with the provided color and the contrast ratio
   * @param {Array} fgColor
   * @param {Number} contrastRatio
   */
  calValidLumaRange(fgColor, contrastRatio) {
    const fgLuma = this.calLuma(fgColor)
    const edgeLuma = (fgLuma + 0.05) / contrastRatio - 0.05
    if (edgeLuma < 1 && edgeLuma > 0) {
      this.minLuma = 0
      this.maxLuma = edgeLuma
    } else if (edgeLuma === 1 || edgeLuma === 0) {
      this.minLuma = this.maxLuma = edgeLuma
    } else {
      this.minLuma = (fgLuma + 0.05) * contrastRatio
      this.maxLuma = 1
    }
  }

  /**
   * Check if generated color's luma is within the valid luma range. If not, launder the color until it is
   * @param {Array} color
   */
  launderColor(color) {
    const luma = this.calLuma(color)
    if (luma >= this.minLuma && luma <= this.maxLuma) {
      return color
    } else {
      const hsl = this._rgb2hsl(color)
      if (luma < this.minLuma) {
        if (this.hash) {
          const step = +(this.minLuma - luma).toFixed(2) || this.step
          hsl[2] === 1 ? hsl[1] = Math.min(hsl[1] + step, 1) : hsl[2] = Math.min(hsl[2] + step, 1)
        } else {
          hsl[2] = (1 - hsl[2]) * Math.random() + hsl[2]
        }
        color = this._hsl2rgb(hsl)
        return this.launderColor(color)
      } else {
        if (this.hash) {
          const step = +(luma - this.maxLuma).toFixed(2) || this.step
          hsl[2] === 0 ? hsl[1] = Math.max(0, hsl[1] - step, 0) : hsl[2] = Math.max(hsl[2] - step, 0)
        } else {
          hsl[2] = Math.random() % hsl[2]
        }
        color = this._hsl2rgb(hsl)
        return this.launderColor(color)
      }
    }
  }
  /**
   * Generates a random color that meets (>=) the contrast ratio. If a string is passed in, generate a consistent contrast-safe color for that string.
   * @param {String} str
   */
  random(str) {
    this.hash = 0
    this.calValidLumaRange(this.color, this.contrast)
    if (this.minLuma === this.maxLuma) {
      this.genColor = this.minLuma ? [255, 255, 255] : [0, 0, 0]
      return
    }
    if (!str) {
      this.genColor = [0, 0, 0].map(v => Math.round(Math.random() * 255))
    } else {
      for (let i = 0; i < str.length; i++) {
        this.hash = str.charCodeAt(i) + ((this.hash << 5) - this.hash)
        this.hash = this.hash & this.hash
      }
      this.genColor = [0, 0, 0]
      this.genColor = this.genColor.map((v, index) => (v = (this.hash >> (index * 8)) & 255))
    }
    this.genColor = this.launderColor(this.genColor)
    return 'rgb(' + this.genColor + ')'
  }
}

export default SafeColor
