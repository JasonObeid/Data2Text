import re
import sys
import codecs
import argparse
import json
from datetime import datetime
from collections import Counter, defaultdict
from text2num import text2num, NumberException
from tokenizer import word_tokenize, sent_tokenize

NUM_PLAYERS = 13
player_name_key = "PLAYER_NAME"
bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
           "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
           "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
           "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
           "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
           "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
           "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])


def _get_player_index(game):
    home_players, vis_players = [], []
    nplayers = len(game["box_score"]["PTS"].keys())
    if game["home_city"] != game["vis_city"]:
        for index in [str(x) for x in range(nplayers)]:
            player_city = game["box_score"]["TEAM_CITY"][index]
            if player_city == game["home_city"]:
                if len(home_players) < NUM_PLAYERS:
                    home_players.append(index)
            else:
                if len(vis_players) < NUM_PLAYERS:
                    vis_players.append(index)
    else:
        for index in range(nplayers):
            if index < nplayers / 2:
                home_players.append(str(index))
            else:
                vis_players.append(str(index))
    return home_players, vis_players


def extract_game_entities(json_data):
    entity_list = []
    for game in json_data:
        entities = {}

        entities[game['home_name']] = 'home TEAM-NAME'
        entities[game["home_line"]["TEAM-NAME"]] = 'home TEAM-NAME'
        entities[game['vis_name']] = 'vis TEAM-NAME'
        entities[game["vis_line"]["TEAM-NAME"]] = 'vis TEAM-NAME'

        entities[game['home_city']] = 'home TEAM-CITY'
        entities[game['vis_city']] = 'vis TEAM-CITY'

        if game["home_city"] == "Los Angeles" or game["home_city"] == 'LA':
            entities['LA'] = 'home TEAM-CITY'
            entities['Los Angeles'] = 'home TEAM-CITY'

        if game["vis_city"] == "Los Angeles" or game["vis_city"] == 'LA':
            entities['LA'] = 'vis TEAM-CITY'
            entities['Los Angeles'] = 'vis TEAM-CITY'

        if game["home_city"] == game["vis_city"]:
            entities['LA'] = 'TEAM-CITY'
            entities['Los Angeles'] = 'TEAM-CITY'

        for player_key in game['box_score']['PLAYER_NAME']:
            player_name = game['box_score']['PLAYER_NAME'][player_key]
            player_first = game['box_score']['FIRST_NAME'][player_key]
            player_second = game['box_score']['SECOND_NAME'][player_key]
            entities[player_name] = player_key
            entities[player_first] = player_key
            entities[player_second] = player_key

        for name in game['box_score']['TEAM_CITY'].values():
            assert name in entities

        for entity in list(entities.keys()):
            parts = entity.strip().split()
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 1 and part not in ["II", "III", "Jr.", "Jr"]:
                        if part not in entities:
                            entities[part] = entities[entity]
                        elif isinstance(entities[part], set):
                            entities[part].add(entities[entity])
                        elif entities[part] != entities[entity]:
                            a_set = set()
                            a_set.add(entities[entity])
                            a_set.add(entities[part])
                            entities[part] = a_set

        result = {}
        for each in entities:
            key = '_'.join(each.split())
            result[key] = entities[each]

        entity_list.append(result)
    return entity_list


def replace_variables(summary_list, game_entity_list, json_data, verbose=False):
    assert len(game_entity_list) == len(json_data) == len(summary_list)

    new_summary_list = []

    for summary, game_entities_orig, game in zip(summary_list, game_entity_list, json_data):
        game_entities = {}
        for k, v in game_entities_orig.items():
            if type(v) != str and len(v) > 1:
                trimmed_name = list(v)[0]
                game_entities[trimmed_name] = k
            else:
                game_entities[v] = k

        new_summary = summary.split(' ')
        home, away = _get_player_index(game)
        home = [int(player) for player in home]
        away = [int(player) for player in away]

        maxPlayerIndex = max(home)

        home_stats, away_stats = get_team_stats(game)

        for idx in range(len(new_summary)):
            if 'var_team' in new_summary[idx]:
                if 'var_team' in new_summary[idx]:
                    teamSide = int(new_summary[idx][9:10])
                    header = new_summary[idx].strip()[12:-1]
                    if teamSide == 0:
                        variable = home_stats[header]
                    else:
                        variable = away_stats[header]
                    new_summary[idx] = variable
            elif 'var_player' in new_summary[idx]:
                indices = re.findall(r"\[(\w+)\]", new_summary[idx])
                teamIndex = int(indices[0])
                playerIndex = int(indices[1])
                header = f'PLAYER-{indices[2]}'
                if teamIndex == 0:
                    player_key = str(min(home) + playerIndex)
                else:
                    player_key = str(playerIndex)
                # if predicted player key doesn't exist, default to max player key
                if int(player_key) > maxPlayerIndex:
                    player_key = str(maxPlayerIndex)
                player_stats = get_player_stats(game, player_key)
                variable = player_stats[header]
                new_summary[idx] = variable
        new_summary_list.append(' '.join(new_summary))
    return new_summary_list


def get_player_stats(game, player_key):
    result = {}
    for key in bs_keys:
        rel_type = key.split('-')[1]
        rel_value = game["box_score"][rel_type][player_key] if player_key is not None else "N/A"
        result[key] = rel_value
    return result


def get_team_stats(game):
    home_result = {}
    vis_result = {}
    for key in ls_keys:
        home_value = game["home_line"][key]
        home_result[key] = home_value

        vis_value = game["vis_line"][key]
        vis_result[key] = vis_value
    return home_result, vis_result


if __name__ == '__main__':
    readme = """
    """
    parser = argparse.ArgumentParser(description=readme)
    parser.add_argument("-d", "--data", required=True, help="rotowire json data")
    parser.add_argument("-o", "--output", required=True, help="output summary path")
    parser.add_argument("-i", "--input", required=True, help="input summary path")
    parser.add_argument('-v', "--verbose", action='store_true', help="verbose")
    args = parser.parse_args()

    json_data = json.load(open(args.data, 'r'))

    with open(args.input, 'r', encoding='utf-8') as inputSummary:
        summary_list = inputSummary.readlines()

    summary_key = 'summary'

    game_entity_list = extract_game_entities(json_data)

    replaced_summary_list = replace_variables(summary_list, game_entity_list, json_data, args.verbose)

    with open(args.output, "w") as outf:
        outf.writelines(replaced_summary_list)
