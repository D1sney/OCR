import os
import re
import io
import contextlib
import datetime
from dateutil.relativedelta import relativedelta

from Bot123_copy import (classify_document, extract_text_from_pdf, extract_system, extract_address, extract_text_from_pdf_all,
                             extract_doc_date_number, check_approval_sheet_in_document, normalize_address_for_comparison, systems_list,
                             check_smeta, extract_sro_date, extract_price_level_explanatory, smeta_stoimost, extract_first_four_digits,
                             check_contract_address, normalize_system, systems_project_required)

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("document_analysis.log")  # Запись в файл
    ]
)

logger = logging.getLogger(__name__)

def document_analysis(file_paths):
    logger.info("Начало анализа документов")
    logger.debug(f"Переданные пути файлов: {file_paths}")
    
    docs = {}
    for file_path in file_paths:
        doc_type = classify_document(file_path)
        if doc_type not in docs:
            docs[doc_type] = file_path
    logger.debug(f"Классифицированные документы: {docs}")

    result_lines = []

    if "Договор" not in docs or "НОПРИЗ" not in docs:
        docs["ЭКУ"] = "Документы по ЭКУ"
        logger.info("Документы классифицированы как ЭКУ")
        result_lines.append("<h3><b>Проверка документов ЭКУ</b></h3>")
        result_lines.append("Получилось! Вот результаты проверки ваших документов ЭКУ:<br>")

        expected_docs = ["Заявление", "Дефектная ведомость", "Акт обследования", "Пояснительная записка", "Смета", "Выписка СРО"]
        missing_docs = [doc for doc in expected_docs if doc not in docs]
        logger.debug("Проверка обязательных документов для ЭКУ")

        if missing_docs:
            result_lines.append(f"<u>Отсутствуют следующие обязательные документы:</u> {', '.join(missing_docs)}<br>")
            logger.warning(f"Отсутствуют документы для ЭКУ: {missing_docs}")
        else:
            result_lines.append("<i>Все обязательные документы присутствуют.</i><br>")
            logger.info("Все обязательные документы ЭКУ присутствуют")

    if "Договор" in docs or "НОПРИЗ" in docs:
        docs["негос"] = "Документы по негос"
        logger.info("Документы классифицированы как негос")
        result_lines.append("<h3><b>Проверка документов негос</b></h3>")
        result_lines.append("Получилось! Вот результаты проверки ваших документов негос:<br>")

        negos_docs = ["НОПРИЗ", "Техническое задание", "Выписка СРО", "Информационно-удостоверяющий лист", "Заявление", "Смета", "Акт обследования", "Дефектная ведомость", "Договор"]
        missing_negos = [doc for doc in negos_docs if doc not in docs]
        logger.debug("Проверка обязательных документов для негос")

        if missing_negos:
            result_lines.append(f"<u>Отсутствуют следующие обязательные документы:</u> {', '.join(missing_negos)}<br>")
            logger.warning(f"Отсутствуют документы для негос: {missing_negos}")
        else:
            result_lines.append("<i>Все обязательные документы присутствуют.</i><br>")
            logger.info("Все обязательные документы негос присутствуют")

    final_errors = []

    if "Заявление" in docs:
        statement_file = docs.get("Заявление")
        st_text = extract_text_from_pdf(statement_file, skip_top_fraction=1/6)
        st_system = extract_system(st_text, systems_list)
        st_address = extract_address(st_text)
        result_lines.append("<h3><b>Заявление</b></h3>")
        result_lines.append(f"<u>Система:</u> {st_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {st_address}<br>")
        logger.info(f"Извлечены данные из Заявления: система={st_system}, адрес={st_address}")

        statement_full_text = extract_text_from_pdf_all(statement_file)
        if "реестр смет" in statement_full_text.lower():
            result_lines.append("<i>В заявлении обнаружен 'Реестр смет': адрес и система совпадают с данными дефектной ведомости.</i><br>")
            logger.info("Обнаружен 'Реестр смет' в Заявлении")
    else:
        result_lines.append("<h3><b>Заявление</b></h3>")
        result_lines.append("<u>Заявление не найдено.</u><br>")
        final_errors.append("Заявление отсутствует.")
        logger.warning("Заявление не найдено")

    if "Дефектная ведомость" in docs:
        defect_file = docs.get("Дефектная ведомость")
        df_text = extract_text_from_pdf(defect_file)
        df_system = extract_system(df_text, systems_list)
        df_address = extract_address(df_text)
        df_doc = extract_doc_date_number(df_text)
        result_lines.append("<h3><b>Дефектная ведомость</b></h3>")
        result_lines.append(f"<u>Система:</u> {df_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {df_address}<br>")
        logger.info(f"Извлечены данные из Дефектной ведомости: система={df_system}, адрес={df_address}")
        if df_doc == "Номер и дата не найдены":
            result_lines.append("<u>Ошибка:</u> Не удалось корректно выгрузить дату и номер. Будет направлена на ручную проверку УТНКР.<br>")
            final_errors.append("Не удалось определить номер и дату дефектной ведомости. УТНКР вручную сравнит номер и дату у документов")
            logger.error("Не удалось извлечь номер и дату из Дефектной ведомости")
        else:
            result_lines.append(f"<u>Номер и дата:</u> {df_doc}<br>")
            logger.debug(f"Извлечены номер и дата Дефектной ведомости: {df_doc}")
        found, preview = check_approval_sheet_in_document(defect_file)
        if found:
            result_lines.append("<i>Лист согласования присутствует.</i><br>")
            logger.info("Лист согласования найден в Дефектной ведомости")
        else:
            result_lines.append("<u>Ошибка:</u> Лист согласования не найден.<br>")
            final_errors.append("Лист согласования отсутствует в дефектной ведомости.")
            logger.warning("Лист согласования отсутствует в Дефектной ведомости")
    else:
        df_doc = "Номер и дата не найдены"
        df_address = "Адрес не найден"
        df_system = "Система не найден"
        result_lines.append("<h3><b>Дефектная ведомость</b></h3>")
        result_lines.append("<u>Дефектная ведомость не найдена.</u><br>")
        final_errors.append("Дефектная ведомость отсутствует.")
        logger.warning("Дефектная ведомость не найдена")

    if "Заявление" in docs and "Дефектная ведомость" in docs:
        result_lines.append("<h3><b>Сравнение Заявления и Дефектной ведомости</b></h3>")
        if normalize_address_for_comparison(st_address) != normalize_address_for_comparison(df_address):
            result_lines.append("<u>Ошибка:</u> Обнаружено различие в адресах:<br>")
            result_lines.append(f"Заявление: {st_address}<br>")
            result_lines.append(f"Дефектная ведомость: {df_address}<br>")
            final_errors.append("Адреса заявления и дефектной ведомости не совпадают.")
            logger.warning(f"Адреса не совпадают: Заявление={st_address}, Дефектная ведомость={df_address}")
        else:
            result_lines.append("<i>Адреса совпадают.</i><br>")
            logger.info("Адреса Заявления и Дефектной ведомости совпадают")
        if st_system.lower() != df_system.lower():
            result_lines.append("<u>Ошибка:</u> Обнаружено различие в системах:<br>")
            result_lines.append(f"Заявление: {st_system}<br>")
            result_lines.append(f"Дефектная ведомость: {df_system}<br>")
            final_errors.append("Системы заявления и дефектной ведомости не совпадают.")
            logger.warning(f"Системы не совпадают: Заявление={st_system}, Дефектная ведомость={df_system}")
        else:
            result_lines.append("<i>Системы совпадают.</i><br>")
            logger.info("Системы Заявления и Дефектной ведомости совпадают")

    if "Смета" in docs:
        smeta_file = docs.get("Смета")
        smeta_output = io.StringIO()
        with contextlib.redirect_stdout(smeta_output):
            smeta_errors = check_smeta(smeta_file, df_address if "Дефектная ведомость" in docs else "",
                                       st_system if "Заявление" in docs else "")
        result_lines.append("<h3><b>Смета</b></h3>")
        result_lines.append(smeta_output.getvalue().replace('\n', '<br>'))
        if smeta_errors:
            final_errors.extend(smeta_errors)
            logger.warning(f"Ошибки в Смете: {smeta_errors}")
        else:
            logger.info("Смета проверена без ошибок")
    else:
        result_lines.append("<u>Смета не найдена.</u><br>")
        final_errors.append("Смета отсутствует.")
        logger.warning("Смета не найдена")

    if "Акт обследования" in docs:
        act_file = docs.get("Акт обследования")
        act_text = extract_text_from_pdf(act_file)
        act_system = extract_system(act_text, systems_list)
        act_address = extract_address(act_text)
        act_doc = extract_doc_date_number(act_text)

        if act_address == "Адрес не найден" and "Дефектная ведомость" in docs:
            act_address = df_address

        result_lines.append("<h3><b>Акт обследования</b></h3>")
        result_lines.append(f"<u>Система:</u> {act_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {act_address}<br>")
        logger.info(f"Извлечены данные из Акта обследования: система={act_system}, адрес={act_address}")

        if act_doc == "Номер и дата не найдены":
            result_lines.append("<u>Ошибка:</u> Не удалось корректно выгрузить дату и номер. Будет направлена на ручную проверку УТНКР.<br>")
            final_errors.append("Не удалось определить номер и дату акта обследования. УТНКР вручную сравнит номер и дату у документов")
            logger.error("Не удалось извлечь номер и дату из Акта обследования")
        else:
            result_lines.append(f"<u>Номер и дата:</u> {act_doc}<br>")
            logger.debug(f"Извлечены номер и дата Акта обследования: {act_doc}")

        found, preview = check_approval_sheet_in_document(act_file)
        if found:
            result_lines.append("<i>Лист согласования присутствует.</i><br>")
            logger.info("Лист согласования найден в Акте обследования")
        else:
            result_lines.append("<u>Ошибка:</u> Лист согласования не найден.<br>")
            final_errors.append("Лист согласования отсутствует в акте обследования.")
            logger.warning("Лист согласования отсутствует в Акте обследования")

        if act_doc != "Номер и дата не найдены" and df_doc != "Номер и дата не найдены":
            if act_doc.lower() == df_doc.lower():
                result_lines.append("<i>Номер и дата совпадают с данными дефектной ведомости.</i><br>")
                logger.info("Номер и дата Акта обследования совпадают с Дефектной ведомостью")
            else:
                result_lines.append("<u>Ошибка:</u> Номер и/или дата НЕ совпадают с данными дефектной ведомости:<br>")
                result_lines.append(f"Акт: {act_doc}<br>")
                result_lines.append(f"Дефектная ведомость: {df_doc}<br>")
                final_errors.append("Номер и/или дата акта обследования не совпадают с дефектной ведомостью.")
                logger.warning(f"Номер/дата не совпадают: Акт={act_doc}, Дефектная ведомость={df_doc}")
        if normalize_address_for_comparison(act_address) == normalize_address_for_comparison(df_address):
            result_lines.append("<i>Адрес совпадает с адресом дефектной ведомости.</i><br>")
            logger.info("Адрес Акта обследования совпадает с Дефектной ведомостью")
        else:
            result_lines.append("<u>Ошибка:</u> Адрес НЕ совпадает с адресом дефектной ведомости:<br>")
            result_lines.append(f"Акт: {act_address}<br>")
            result_lines.append(f"Дефектная ведомость: {df_address}<br>")
            final_errors.append("Адрес акта обследования не совпадает с адресом дефектной ведомости.")
            logger.warning(f"Адреса не совпадают: Акт={act_address}, Дефектная ведомость={df_address}")
        if act_system.lower() == df_system.lower():
            result_lines.append("<i>Система совпадает с системой дефектной ведомости.</i><br>")
            logger.info("Система Акта обследования совпадает с Дефектной ведомостью")
        else:
            result_lines.append("<u>Ошибка:</u> Система НЕ совпадает с системой дефектной ведомости:<br>")
            result_lines.append(f"Акт: {act_system}<br>")
            result_lines.append(f"Дефектная ведомость: {df_system}<br>")
            final_errors.append("Система акта обследования не совпадает с системой дефектной ведомости.")
            logger.warning(f"Системы не совпадают: Акт={act_system}, Дефектная ведомость={df_system}")
    else:
        result_lines.append("<h3><b>Акт обследования</b></h3>")
        result_lines.append("<u>Акт обследования не найден.</u><br>")
        final_errors.append("Акт обследования отсутствует.")
        logger.warning("Акт обследования не найден")

    if "Техническое задание" in docs:
        tech_file = docs.get("Техническое задание")
        tech_text = extract_text_from_pdf(tech_file)
        tech_system = extract_system(tech_text, systems_list)
        tech_address = extract_address(tech_text)
        tech_doc = extract_doc_date_number(tech_text)

        if tech_address == "Адрес не найден":
            tech_address = df_address

        result_lines.append("<h3><b>Техническое задание</b></h3>")
        result_lines.append(f"<u>Система:</u> {tech_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {tech_address}<br>")
        logger.info(f"Извлечены данные из Технического задания: система={tech_system}, адрес={tech_address}")

        if tech_doc == "Дата не найдена":
            result_lines.append("<u>Ошибка:</u> Не удалось корректно выгрузить дату. Будет направлена на ручную проверку УТНКР.<br>")
            final_errors.append("Не удалось определить дату технического задания. УТНКР вручную сравнит номер и дату у документов")
            logger.error("Не удалось извлечь дату из Технического задания")
        else:
            result_lines.append(f"<u>Дата:</u> {tech_doc}<br>")
            logger.debug(f"Извлечена дата Технического задания: {tech_doc}")

    if "Выписка СРО" in docs:
        sro_file = docs.get("Выписка СРО")
        sro_text = extract_text_from_pdf(sro_file)
        sro_date = extract_sro_date(sro_text)

        result_lines.append("<h3><b>Выписка СРО</b></h3>")
        if sro_date:
            current_date = datetime.date.today()
            deadline_date = current_date + datetime.timedelta(days=30)
            if sro_date <= deadline_date:
                result_lines.append(f"<i>Выписка СРО оформлена вовремя:</i> {sro_date} (срок до {deadline_date})<br>")
                logger.info(f"Выписка СРО оформлена вовремя: {sro_date}")
            else:
                result_lines.append(f"<u>Ошибка:</u> Выписка СРО оформлена с опозданием: {sro_date}. Допустимый срок - до {deadline_date}<br>")
                logger.warning(f"Выписка СРО просрочена: {sro_date}, срок до {deadline_date}")
        else:
            result_lines.append("<u>Ошибка:</u> Дата выписки СРО не найдена в документе<br>")
            logger.warning("Дата выписки СРО не найдена")

    if "Пояснительная записка" in docs:
        explanatory_file = docs.get("Пояснительная записка")
        expl_text = extract_text_from_pdf_all(explanatory_file)
        expl_system = extract_system(expl_text, systems_list)
        expl_address = extract_address(expl_text)
        expl_price_date, expl_price_display = extract_price_level_explanatory(expl_text)

        expl_smet_match = re.search(
            r'Сметная стоимость.*?([\d\s.,]+)\s*тыс\.?\s*руб',
            expl_text,
            re.IGNORECASE | re.DOTALL
        )
        result_lines.append("<h3><b>Пояснительная записка</b></h3>")
        result_lines.append(f"<u>Система:</u> {expl_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {expl_address}<br>")
        logger.info(f"Извлечены данные из Пояснительной записки: система={expl_system}, адрес={expl_address}")
        if expl_smet_match:
            expl_stoimost = expl_smet_match.group(1).replace(" ", "")
            result_lines.append(f"<u>Сметная стоимость:</u> {expl_stoimost} тыс. руб.<br>")
            logger.debug(f"Извлечена сметная стоимость из Пояснительной записки: {expl_stoimost}")
        else:
            expl_stoimost = None

        if expl_stoimost is None:
            result_lines.append("<u>Ошибка:</u> Сметная стоимость отсутствует, сравнение невозможно.<br>")
            final_errors.append("Сметная стоимость отсутствует в пояснительной записке.")
            logger.warning("Сметная стоимость отсутствует в Пояснительной записке")
        elif smeta_stoimost is None:
            result_lines.append("<u>Ошибка:</u> Сметная стоимость в смете отсутствует, сравнение невозможно.<br>")
            final_errors.append("Сметная стоимость отсутствует в смете.")
            logger.warning("Сметная стоимость отсутствует в Смете")
        else:
            expl_cost_4 = extract_first_four_digits(expl_stoimost)
            smeta_cost_4 = extract_first_four_digits(smeta_stoimost)
            if expl_cost_4 == smeta_cost_4:
                result_lines.append("<i>Сметная стоимость соответствует смете.</i><br>")
                logger.info("Сметная стоимость в Пояснительной записке соответствует Смете")
            else:
                result_lines.append("<u>Ошибка:</u> Сметная стоимость не соответствует смете.<br>")
                final_errors.append("Сметная стоимость не соответствует смете.")
                logger.warning("Сметная стоимость в Пояснительной записке не соответствует Смете")

        if expl_price_display:
            result_lines.append(f"<u>Дата уровня цен:</u> {expl_price_display}<br>")
            logger.debug(f"Извлечена дата уровня цен: {expl_price_display}")
        else:
            result_lines.append("<u>Ошибка:</u> Не удалось распознать уровень цен.<br>")
            final_errors.append("Не удалось распознать уровень цен в пояснительной записке.")
            logger.error("Не удалось распознать уровень цен в Пояснительной записке")

        if normalize_address_for_comparison(expl_address) == normalize_address_for_comparison(df_address):
            result_lines.append("<i>Адрес совпадает с адресом дефектной ведомости.</i><br>")
            logger.info("Адрес Пояснительной записки совпадает с Дефектной ведомостью")
        else:
            result_lines.append("<u>Ошибка:</u> Адрес НЕ совпадает с адресом дефектной ведомости:<br>")
            result_lines.append(f"Пояснительная: {expl_address}<br>")
            result_lines.append(f"Дефектная ведомость: {df_address}<br>")
            final_errors.append("Адрес пояснительной записки не совпадает с адресом дефектной ведомости.")
            logger.warning(f"Адреса не совпадают: Пояснительная={expl_address}, Дефектная ведомость={df_address}")

        if expl_system.lower() == df_system.lower():
            result_lines.append("<i>Система совпадает с системой дефектной ведомости.</i><br>")
            logger.info("Система Пояснительной записки совпадает с Дефектной ведомостью")
        else:
            result_lines.append("<u>Ошибка:</u> Система НЕ совпадает с системой дефектной ведомости:<br>")
            result_lines.append(f"Пояснительная: {expl_system}<br>")
            result_lines.append(f"Дефектная ведомость: {df_system}<br>")
            final_errors.append("Система пояснительной записки не совпадает с системой дефектной ведомости.")
            logger.warning(f"Системы не совпадают: Пояснительная={expl_system}, Дефектная ведомость={df_system}")

        if expl_price_date:
            today = datetime.date.today()
            if expl_price_date >= today - relativedelta(months=3):
                result_lines.append("<i>Уровень цен актуален (не старше 3 месяцев).</i><br>")
                logger.info("Уровень цен в Пояснительной записке актуален")
            else:
                result_lines.append("<u>Ошибка:</u> Уровень цен устарел!<br>")
                final_errors.append("Уровень цен в пояснительной записке устарел.")
                logger.warning("Уровень цен в Пояснительной записке устарел")
    else:
        result_lines.append("<h3><b>Пояснительная записка</b></h3>")
        result_lines.append("<u>Пояснительная записка не найдена.</u><br>")
        final_errors.append("Пояснительная записка отсутствует.")
        logger.warning("Пояснительная записка не найдена")

    if "Договор" in docs:
        dog_file = docs.get("Договор")
        dog_address = check_contract_address(dog_file, st_address)
        result_lines.append("<h3><b>Договор</b></h3>")
        if dog_address:
            result_lines.append(f"<u>Извлечённый адрес:</u> {dog_address}<br>")
            st_address = dog_address
            logger.info(f"Извлечен адрес из Договора: {dog_address}")
        else:
            result_lines.append("<i>Адрес не найден, используется адрес из заявления.</i><br>")
            logger.debug("Адрес в Договоре не найден")

    if ("Заявление" in docs) and (normalize_system(st_system) in [normalize_system(x) for x in systems_project_required]):
        project_file = docs.get("Проект")
        conclusion_file = docs.get("Заключение")
        valid_project = False
        valid_conclusion = False
        proj_text = ""
        concl_text = ""

        if project_file and os.path.splitext(project_file)[1].lower() == ".pdf":
            proj_text = extract_text_from_pdf(project_file)
            if "проектная документация" in proj_text.lower():
                valid_project = True
        if conclusion_file and os.path.splitext(conclusion_file)[1].lower() == ".pdf":
            concl_text = extract_text_from_pdf(conclusion_file)
            if "заключение" in concl_text.lower():
                valid_conclusion = True

        result_lines.append("<h3><b>Проверка Проекта и Заключения</b></h3>")
        if valid_project and valid_conclusion:
            result_lines.append("<u>Проект:</u><br>")
            result_lines.append(f"{proj_text}<br>")
            result_lines.append("<u>Заключение:</u><br>")
            result_lines.append(f"{concl_text}<br>")
            logger.info("Проект и Заключение найдены и корректны")
        else:
            missing = []
            if not valid_project:
                missing.append("проект")
            if not valid_conclusion:
                missing.append("заключение")
            result_lines.append(f"<u>Ошибка:</u> Для данной системы требуется проект и заключение, но следующие документы не найдены или некорректны: {', '.join(missing)}.<br>")
            final_errors.append(f"Отсутствуют или некорректны: {', '.join(missing)}")
            logger.warning(f"Отсутствуют или некорректны документы для системы: {missing}")

    result_lines.append("<h3><b>Итог</b></h3>")
    if final_errors:
        result_lines.append("<u>Обнаружены следующие ошибки:</u><ul>")
        for err in final_errors:
            result_lines.append(f"<li>{err}</li>")
        result_lines.append("</ul>")
        logger.warning(f"Обнаружены ошибки в документах: {final_errors}")
    else:
        result_lines.append("<i>Ошибок не обнаружено. Все документы соответствуют требованиям.</i>")
        logger.info("Анализ завершен без ошибок")

    final_result = "".join(result_lines)
    logger.info("Анализ документов завершен")
    return final_result