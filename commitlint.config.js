module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Allowed commit types
    'type-enum': [
      2, 'always',
      ['feat', 'fix', 'perf', 'refactor', 'chore', 'docs', 'ci', 'test']
    ],

    'type-empty':      [2, 'never'],
    'subject-empty':   [2, 'never'],
    'subject-min-length': [2, 'always', 10],
    'subject-full-stop':  [2, 'never', '.'],
    'type-case':          [2, 'always', 'lower-case'],
    'body-max-line-length': [2, 'always', 100],
  },

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
};
