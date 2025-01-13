# SafeColor
SafeColor generates accessible colors that complies with WCAG success criteria 1.4.3 (or any contrast ratio of your choice).
It can be used to:

1. generate a random color that is contrast safe with a given color
2. generate a consistent, contrast safe color for a string

No need to worry about your base color is light/dark or for foreground/background. If the given color is too light to meet your desired contrast ratio, SafeColor will look for a darker color and vise versa.


## Install

`npm install safecolor`

## Usage

`import SafeColor from 'safecolor'`

### Basic

This will assume that the generated color should be contrast safe (>= AA standard: 4.5) with black(rgb(0, 0, 0))

```javascript
  safeColor = new SafeColor()

  safeColor.random()
  // >> rgb(104, 145, 26)
  // contrast ratio = 5.65

  safeColor.random('hello world')
  // >> rgb(196,226,239)
  // contrast ratio = 15.47
```
### With options

```javascript
  safeColor = new SafeColor({
    color: [255, 255, 255], // 8bit RGB value in array [r, g, b]
    contrast: 4.5,  // the contrast ratio between the option color and the generated color will >= this
  })

  safeColor.random()
  // >> rgb(32,80,46)
  // contrast ratio = 9.34

  safeColor.random('hello world')
  // >> rgb(20,57,74)
  // contrast ratio = 12.25
```

## Options

**color**

- type: `Array`
- default: `[0, 0, 0]`

**contrast**

- type: `Number`
- default: `4.5`

## Notice
ES6 features: destructing assignment and map are used in this script. You may need polyfill for the script to work properly.

Note: to keep this as simple as possible, the output is a RGB value in string. If any built-in conversions (to HEX, to HSL) will make SafeColor much more convenient for you, please contact me to add the feature or feel free to pull request. Cheers!





