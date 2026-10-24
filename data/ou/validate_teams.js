#!/usr/bin/env node
/*
 * data/ou/validate_teams.js
 * Validates many teams in-process with Showdown's own TeamValidator.
 *
 * Usage: node validate_teams.js <showdown-dir> [format] < teams.jsonl > results.jsonl
 * Input lines:  {"id": "...", "team": "<Showdown export text>"}
 * Output lines: {"id", "valid", "problems", "packed", "export", "sets"}
 *   sets are Showdown's parsed PokemonSet objects (canonical names), packed/export are
 *   Teams.pack / Teams.export of those sets.
 */
'use strict';

const path = require('path');
const readline = require('readline');

const showdownDir = process.argv[2];
const format = process.argv[3] || 'gen9ou';
if (!showdownDir) {
	console.error('usage: node validate_teams.js <showdown-dir> [format] < teams.jsonl');
	process.exit(2);
}

const {Teams, TeamValidator} = require(path.join(path.resolve(showdownDir), 'dist', 'sim'));
const validator = TeamValidator.get(format);

function check(team) {
	let sets;
	try {
		sets = Teams.import(team);
	} catch (e) {
		return {valid: false, problems: [`import failed: ${e.message}`]};
	}
	if (!sets || !sets.length) return {valid: false, problems: ['import failed: no sets parsed']};
	let problems;
	try {
		problems = validator.validateTeam(sets);
	} catch (e) {
		problems = [`validator crashed: ${e.message}`];
	}
	return {
		valid: !problems,
		problems: problems || [],
		packed: Teams.pack(sets),
		export: Teams.export(sets),
		sets,
	};
}

const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
rl.on('line', line => {
	if (!line.trim()) return;
	const {id, team} = JSON.parse(line);
	process.stdout.write(JSON.stringify({id, ...check(team)}) + '\n');
});
