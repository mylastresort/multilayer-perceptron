// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Allowed commit types — only these pass
    'type-enum': [
      2, 'always',
      ['feat', 'fix', 'perf', 'refactor', 'chore', 'docs', 'ci', 'test']
    ],

    // Type is mandatory
    'type-empty': [2, 'never'],

    // Subject (description) must exist
    'subject-empty': [2, 'never'],

    // Subject must be at least 10 chars — forces meaningful messages
    'subject-min-length': [2, 'always', 10],

    // No period at end of subject
    'subject-full-stop': [2, 'never', '.'],

    // Lowercase type
    'type-case': [2, 'always', 'lower-case'],

    // Body line length (for long descriptions)
    'body-max-line-length': [2, 'always', 100],
  },

  // Custom plugin: perf commits must contain a number (the metric)
  plugins: [
    {
      rules: {
        'perf-requires-metric': ({ type, subject }) => {
          if (type !== 'perf') return [true];
          const hasMetric = /\d/.test(subject);
          return [
            hasMetric,
            'perf commits must include a measurable metric (e.g. "reduce p99 320ms → 48ms")'
          ];
        },
      },
    },
  ],

  rules: {
    'perf-requires-metric': [2, 'always'],
  },
};
