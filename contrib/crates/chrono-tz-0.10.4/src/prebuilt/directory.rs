pub const IANA_TZDB_VERSION: &str = "2025b";

use crate::timezones::Tz;

pub const CET: Tz = Tz::CET;
pub const CST6CDT: Tz = Tz::CST6CDT;
pub const Cuba: Tz = Tz::Cuba;
pub const EET: Tz = Tz::EET;
pub const EST: Tz = Tz::EST;
pub const EST5EDT: Tz = Tz::EST5EDT;
pub const Egypt: Tz = Tz::Egypt;
pub const Eire: Tz = Tz::Eire;
pub const GB: Tz = Tz::GB;
pub const GBEire: Tz = Tz::GBEire;
pub const GMT: Tz = Tz::GMT;
pub const GMTPlus0: Tz = Tz::GMTPlus0;
pub const GMTMinus0: Tz = Tz::GMTMinus0;
pub const GMT0: Tz = Tz::GMT0;
pub const Greenwich: Tz = Tz::Greenwich;
pub const HST: Tz = Tz::HST;
pub const Hongkong: Tz = Tz::Hongkong;
pub const Iceland: Tz = Tz::Iceland;
pub const Iran: Tz = Tz::Iran;
pub const Israel: Tz = Tz::Israel;
pub const Jamaica: Tz = Tz::Jamaica;
pub const Japan: Tz = Tz::Japan;
pub const Kwajalein: Tz = Tz::Kwajalein;
pub const Libya: Tz = Tz::Libya;
pub const MET: Tz = Tz::MET;
pub const MST: Tz = Tz::MST;
pub const MST7MDT: Tz = Tz::MST7MDT;
pub const NZ: Tz = Tz::NZ;
pub const NZCHAT: Tz = Tz::NZCHAT;
pub const Navajo: Tz = Tz::Navajo;
pub const PRC: Tz = Tz::PRC;
pub const PST8PDT: Tz = Tz::PST8PDT;
pub const Poland: Tz = Tz::Poland;
pub const Portugal: Tz = Tz::Portugal;
pub const ROC: Tz = Tz::ROC;
pub const ROK: Tz = Tz::ROK;
pub const Singapore: Tz = Tz::Singapore;
pub const Turkey: Tz = Tz::Turkey;
pub const UCT: Tz = Tz::UCT;
pub const UTC: Tz = Tz::UTC;
pub const Universal: Tz = Tz::Universal;
pub const WSU: Tz = Tz::WSU;
pub const WET: Tz = Tz::WET;
pub const Zulu: Tz = Tz::Zulu;

pub mod Africa {
    use crate::timezones::Tz;

    pub const Abidjan: Tz = Tz::Africa__Abidjan;
    pub const Accra: Tz = Tz::Africa__Accra;
    pub const Addis_Ababa: Tz = Tz::Africa__Addis_Ababa;
    pub const Algiers: Tz = Tz::Africa__Algiers;
    pub const Asmara: Tz = Tz::Africa__Asmara;
    pub const Asmera: Tz = Tz::Africa__Asmera;
    pub const Bamako: Tz = Tz::Africa__Bamako;
    pub const Bangui: Tz = Tz::Africa__Bangui;
    pub const Banjul: Tz = Tz::Africa__Banjul;
    pub const Bissau: Tz = Tz::Africa__Bissau;
    pub const Blantyre: Tz = Tz::Africa__Blantyre;
    pub const Brazzaville: Tz = Tz::Africa__Brazzaville;
    pub const Bujumbura: Tz = Tz::Africa__Bujumbura;
    pub const Cairo: Tz = Tz::Africa__Cairo;
    pub const Casablanca: Tz = Tz::Africa__Casablanca;
    pub const Ceuta: Tz = Tz::Africa__Ceuta;
    pub const Conakry: Tz = Tz::Africa__Conakry;
    pub const Dakar: Tz = Tz::Africa__Dakar;
    pub const Dar_es_Salaam: Tz = Tz::Africa__Dar_es_Salaam;
    pub const Djibouti: Tz = Tz::Africa__Djibouti;
    pub const Douala: Tz = Tz::Africa__Douala;
    pub const El_Aaiun: Tz = Tz::Africa__El_Aaiun;
    pub const Freetown: Tz = Tz::Africa__Freetown;
    pub const Gaborone: Tz = Tz::Africa__Gaborone;
    pub const Harare: Tz = Tz::Africa__Harare;
    pub const Johannesburg: Tz = Tz::Africa__Johannesburg;
    pub const Juba: Tz = Tz::Africa__Juba;
    pub const Kampala: Tz = Tz::Africa__Kampala;
    pub const Khartoum: Tz = Tz::Africa__Khartoum;
    pub const Kigali: Tz = Tz::Africa__Kigali;
    pub const Kinshasa: Tz = Tz::Africa__Kinshasa;
    pub const Lagos: Tz = Tz::Africa__Lagos;
    pub const Libreville: Tz = Tz::Africa__Libreville;
    pub const Lome: Tz = Tz::Africa__Lome;
    pub const Luanda: Tz = Tz::Africa__Luanda;
    pub const Lubumbashi: Tz = Tz::Africa__Lubumbashi;
    pub const Lusaka: Tz = Tz::Africa__Lusaka;
    pub const Malabo: Tz = Tz::Africa__Malabo;
    pub const Maputo: Tz = Tz::Africa__Maputo;
    pub const Maseru: Tz = Tz::Africa__Maseru;
    pub const Mbabane: Tz = Tz::Africa__Mbabane;
    pub const Mogadishu: Tz = Tz::Africa__Mogadishu;
    pub const Monrovia: Tz = Tz::Africa__Monrovia;
    pub const Nairobi: Tz = Tz::Africa__Nairobi;
    pub const Ndjamena: Tz = Tz::Africa__Ndjamena;
    pub const Niamey: Tz = Tz::Africa__Niamey;
    pub const Nouakchott: Tz = Tz::Africa__Nouakchott;
    pub const Ouagadougou: Tz = Tz::Africa__Ouagadougou;
    pub const PortoNovo: Tz = Tz::Africa__PortoNovo;
    pub const Sao_Tome: Tz = Tz::Africa__Sao_Tome;
    pub const Timbuktu: Tz = Tz::Africa__Timbuktu;
    pub const Tripoli: Tz = Tz::Africa__Tripoli;
    pub const Tunis: Tz = Tz::Africa__Tunis;
    pub const Windhoek: Tz = Tz::Africa__Windhoek;
}

pub mod America {
    use crate::timezones::Tz;

    pub mod Argentina {
        use crate::timezones::Tz;

        pub const Buenos_Aires: Tz = Tz::America__Argentina__Buenos_Aires;
        pub const Catamarca: Tz = Tz::America__Argentina__Catamarca;
        pub const ComodRivadavia: Tz = Tz::America__Argentina__ComodRivadavia;
        pub const Cordoba: Tz = Tz::America__Argentina__Cordoba;
        pub const Jujuy: Tz = Tz::America__Argentina__Jujuy;
        pub const La_Rioja: Tz = Tz::America__Argentina__La_Rioja;
        pub const Mendoza: Tz = Tz::America__Argentina__Mendoza;
        pub const Rio_Gallegos: Tz = Tz::America__Argentina__Rio_Gallegos;
        pub const Salta: Tz = Tz::America__Argentina__Salta;
        pub const San_Juan: Tz = Tz::America__Argentina__San_Juan;
        pub const San_Luis: Tz = Tz::America__Argentina__San_Luis;
        pub const Tucuman: Tz = Tz::America__Argentina__Tucuman;
        pub const Ushuaia: Tz = Tz::America__Argentina__Ushuaia;
    }

    pub mod Indiana {
        use crate::timezones::Tz;

        pub const Indianapolis: Tz = Tz::America__Indiana__Indianapolis;
        pub const Knox: Tz = Tz::America__Indiana__Knox;
        pub const Marengo: Tz = Tz::America__Indiana__Marengo;
        pub const Petersburg: Tz = Tz::America__Indiana__Petersburg;
        pub const Tell_City: Tz = Tz::America__Indiana__Tell_City;
        pub const Vevay: Tz = Tz::America__Indiana__Vevay;
        pub const Vincennes: Tz = Tz::America__Indiana__Vincennes;
        pub const Winamac: Tz = Tz::America__Indiana__Winamac;
    }

    pub mod Kentucky {
        use crate::timezones::Tz;

        pub const Louisville: Tz = Tz::America__Kentucky__Louisville;
        pub const Monticello: Tz = Tz::America__Kentucky__Monticello;
    }

    pub mod North_Dakota {
        use crate::timezones::Tz;

        pub const Beulah: Tz = Tz::America__North_Dakota__Beulah;
        pub const Center: Tz = Tz::America__North_Dakota__Center;
        pub const New_Salem: Tz = Tz::America__North_Dakota__New_Salem;
    }

    pub const Adak: Tz = Tz::America__Adak;
    pub const Anchorage: Tz = Tz::America__Anchorage;
    pub const Anguilla: Tz = Tz::America__Anguilla;
    pub const Antigua: Tz = Tz::America__Antigua;
    pub const Araguaina: Tz = Tz::America__Araguaina;
    pub const Aruba: Tz = Tz::America__Aruba;
    pub const Asuncion: Tz = Tz::America__Asuncion;
    pub const Atikokan: Tz = Tz::America__Atikokan;
    pub const Atka: Tz = Tz::America__Atka;
    pub const Bahia: Tz = Tz::America__Bahia;
    pub const Bahia_Banderas: Tz = Tz::America__Bahia_Banderas;
    pub const Barbados: Tz = Tz::America__Barbados;
    pub const Belem: Tz = Tz::America__Belem;
    pub const Belize: Tz = Tz::America__Belize;
    pub const BlancSablon: Tz = Tz::America__BlancSablon;
    pub const Boa_Vista: Tz = Tz::America__Boa_Vista;
    pub const Bogota: Tz = Tz::America__Bogota;
    pub const Boise: Tz = Tz::America__Boise;
    pub const Buenos_Aires: Tz = Tz::America__Buenos_Aires;
    pub const Cambridge_Bay: Tz = Tz::America__Cambridge_Bay;
    pub const Campo_Grande: Tz = Tz::America__Campo_Grande;
    pub const Cancun: Tz = Tz::America__Cancun;
    pub const Caracas: Tz = Tz::America__Caracas;
    pub const Catamarca: Tz = Tz::America__Catamarca;
    pub const Cayenne: Tz = Tz::America__Cayenne;
    pub const Cayman: Tz = Tz::America__Cayman;
    pub const Chicago: Tz = Tz::America__Chicago;
    pub const Chihuahua: Tz = Tz::America__Chihuahua;
    pub const Ciudad_Juarez: Tz = Tz::America__Ciudad_Juarez;
    pub const Coral_Harbour: Tz = Tz::America__Coral_Harbour;
    pub const Cordoba: Tz = Tz::America__Cordoba;
    pub const Costa_Rica: Tz = Tz::America__Costa_Rica;
    pub const Coyhaique: Tz = Tz::America__Coyhaique;
    pub const Creston: Tz = Tz::America__Creston;
    pub const Cuiaba: Tz = Tz::America__Cuiaba;
    pub const Curacao: Tz = Tz::America__Curacao;
    pub const Danmarkshavn: Tz = Tz::America__Danmarkshavn;
    pub const Dawson: Tz = Tz::America__Dawson;
    pub const Dawson_Creek: Tz = Tz::America__Dawson_Creek;
    pub const Denver: Tz = Tz::America__Denver;
    pub const Detroit: Tz = Tz::America__Detroit;
    pub const Dominica: Tz = Tz::America__Dominica;
    pub const Edmonton: Tz = Tz::America__Edmonton;
    pub const Eirunepe: Tz = Tz::America__Eirunepe;
    pub const El_Salvador: Tz = Tz::America__El_Salvador;
    pub const Ensenada: Tz = Tz::America__Ensenada;
    pub const Fort_Nelson: Tz = Tz::America__Fort_Nelson;
    pub const Fort_Wayne: Tz = Tz::America__Fort_Wayne;
    pub const Fortaleza: Tz = Tz::America__Fortaleza;
    pub const Glace_Bay: Tz = Tz::America__Glace_Bay;
    pub const Godthab: Tz = Tz::America__Godthab;
    pub const Goose_Bay: Tz = Tz::America__Goose_Bay;
    pub const Grand_Turk: Tz = Tz::America__Grand_Turk;
    pub const Grenada: Tz = Tz::America__Grenada;
    pub const Guadeloupe: Tz = Tz::America__Guadeloupe;
    pub const Guatemala: Tz = Tz::America__Guatemala;
    pub const Guayaquil: Tz = Tz::America__Guayaquil;
    pub const Guyana: Tz = Tz::America__Guyana;
    pub const Halifax: Tz = Tz::America__Halifax;
    pub const Havana: Tz = Tz::America__Havana;
    pub const Hermosillo: Tz = Tz::America__Hermosillo;
    pub const Indianapolis: Tz = Tz::America__Indianapolis;
    pub const Inuvik: Tz = Tz::America__Inuvik;
    pub const Iqaluit: Tz = Tz::America__Iqaluit;
    pub const Jamaica: Tz = Tz::America__Jamaica;
    pub const Jujuy: Tz = Tz::America__Jujuy;
    pub const Juneau: Tz = Tz::America__Juneau;
    pub const Knox_IN: Tz = Tz::America__Knox_IN;
    pub const Kralendijk: Tz = Tz::America__Kralendijk;
    pub const La_Paz: Tz = Tz::America__La_Paz;
    pub const Lima: Tz = Tz::America__Lima;
    pub const Los_Angeles: Tz = Tz::America__Los_Angeles;
    pub const Louisville: Tz = Tz::America__Louisville;
    pub const Lower_Princes: Tz = Tz::America__Lower_Princes;
    pub const Maceio: Tz = Tz::America__Maceio;
    pub const Managua: Tz = Tz::America__Managua;
    pub const Manaus: Tz = Tz::America__Manaus;
    pub const Marigot: Tz = Tz::America__Marigot;
    pub const Martinique: Tz = Tz::America__Martinique;
    pub const Matamoros: Tz = Tz::America__Matamoros;
    pub const Mazatlan: Tz = Tz::America__Mazatlan;
    pub const Mendoza: Tz = Tz::America__Mendoza;
    pub const Menominee: Tz = Tz::America__Menominee;
    pub const Merida: Tz = Tz::America__Merida;
    pub const Metlakatla: Tz = Tz::America__Metlakatla;
    pub const Mexico_City: Tz = Tz::America__Mexico_City;
    pub const Miquelon: Tz = Tz::America__Miquelon;
    pub const Moncton: Tz = Tz::America__Moncton;
    pub const Monterrey: Tz = Tz::America__Monterrey;
    pub const Montevideo: Tz = Tz::America__Montevideo;
    pub const Montreal: Tz = Tz::America__Montreal;
    pub const Montserrat: Tz = Tz::America__Montserrat;
    pub const Nassau: Tz = Tz::America__Nassau;
    pub const New_York: Tz = Tz::America__New_York;
    pub const Nipigon: Tz = Tz::America__Nipigon;
    pub const Nome: Tz = Tz::America__Nome;
    pub const Noronha: Tz = Tz::America__Noronha;
    pub const Nuuk: Tz = Tz::America__Nuuk;
    pub const Ojinaga: Tz = Tz::America__Ojinaga;
    pub const Panama: Tz = Tz::America__Panama;
    pub const Pangnirtung: Tz = Tz::America__Pangnirtung;
    pub const Paramaribo: Tz = Tz::America__Paramaribo;
    pub const Phoenix: Tz = Tz::America__Phoenix;
    pub const PortauPrince: Tz = Tz::America__PortauPrince;
    pub const Port_of_Spain: Tz = Tz::America__Port_of_Spain;
    pub const Porto_Acre: Tz = Tz::America__Porto_Acre;
    pub const Porto_Velho: Tz = Tz::America__Porto_Velho;
    pub const Puerto_Rico: Tz = Tz::America__Puerto_Rico;
    pub const Punta_Arenas: Tz = Tz::America__Punta_Arenas;
    pub const Rainy_River: Tz = Tz::America__Rainy_River;
    pub const Rankin_Inlet: Tz = Tz::America__Rankin_Inlet;
    pub const Recife: Tz = Tz::America__Recife;
    pub const Regina: Tz = Tz::America__Regina;
    pub const Resolute: Tz = Tz::America__Resolute;
    pub const Rio_Branco: Tz = Tz::America__Rio_Branco;
    pub const Rosario: Tz = Tz::America__Rosario;
    pub const Santa_Isabel: Tz = Tz::America__Santa_Isabel;
    pub const Santarem: Tz = Tz::America__Santarem;
    pub const Santiago: Tz = Tz::America__Santiago;
    pub const Santo_Domingo: Tz = Tz::America__Santo_Domingo;
    pub const Sao_Paulo: Tz = Tz::America__Sao_Paulo;
    pub const Scoresbysund: Tz = Tz::America__Scoresbysund;
    pub const Shiprock: Tz = Tz::America__Shiprock;
    pub const Sitka: Tz = Tz::America__Sitka;
    pub const St_Barthelemy: Tz = Tz::America__St_Barthelemy;
    pub const St_Johns: Tz = Tz::America__St_Johns;
    pub const St_Kitts: Tz = Tz::America__St_Kitts;
    pub const St_Lucia: Tz = Tz::America__St_Lucia;
    pub const St_Thomas: Tz = Tz::America__St_Thomas;
    pub const St_Vincent: Tz = Tz::America__St_Vincent;
    pub const Swift_Current: Tz = Tz::America__Swift_Current;
    pub const Tegucigalpa: Tz = Tz::America__Tegucigalpa;
    pub const Thule: Tz = Tz::America__Thule;
    pub const Thunder_Bay: Tz = Tz::America__Thunder_Bay;
    pub const Tijuana: Tz = Tz::America__Tijuana;
    pub const Toronto: Tz = Tz::America__Toronto;
    pub const Tortola: Tz = Tz::America__Tortola;
    pub const Vancouver: Tz = Tz::America__Vancouver;
    pub const Virgin: Tz = Tz::America__Virgin;
    pub const Whitehorse: Tz = Tz::America__Whitehorse;
    pub const Winnipeg: Tz = Tz::America__Winnipeg;
    pub const Yakutat: Tz = Tz::America__Yakutat;
    pub const Yellowknife: Tz = Tz::America__Yellowknife;
}

pub mod Antarctica {
    use crate::timezones::Tz;

    pub const Casey: Tz = Tz::Antarctica__Casey;
    pub const Davis: Tz = Tz::Antarctica__Davis;
    pub const DumontDUrville: Tz = Tz::Antarctica__DumontDUrville;
    pub const Macquarie: Tz = Tz::Antarctica__Macquarie;
    pub const Mawson: Tz = Tz::Antarctica__Mawson;
    pub const McMurdo: Tz = Tz::Antarctica__McMurdo;
    pub const Palmer: Tz = Tz::Antarctica__Palmer;
    pub const Rothera: Tz = Tz::Antarctica__Rothera;
    pub const South_Pole: Tz = Tz::Antarctica__South_Pole;
    pub const Syowa: Tz = Tz::Antarctica__Syowa;
    pub const Troll: Tz = Tz::Antarctica__Troll;
    pub const Vostok: Tz = Tz::Antarctica__Vostok;
}

pub mod Arctic {
    use crate::timezones::Tz;

    pub const Longyearbyen: Tz = Tz::Arctic__Longyearbyen;
}

pub mod Asia {
    use crate::timezones::Tz;

    pub const Aden: Tz = Tz::Asia__Aden;
    pub const Almaty: Tz = Tz::Asia__Almaty;
    pub const Amman: Tz = Tz::Asia__Amman;
    pub const Anadyr: Tz = Tz::Asia__Anadyr;
    pub const Aqtau: Tz = Tz::Asia__Aqtau;
    pub const Aqtobe: Tz = Tz::Asia__Aqtobe;
    pub const Ashgabat: Tz = Tz::Asia__Ashgabat;
    pub const Ashkhabad: Tz = Tz::Asia__Ashkhabad;
    pub const Atyrau: Tz = Tz::Asia__Atyrau;
    pub const Baghdad: Tz = Tz::Asia__Baghdad;
    pub const Bahrain: Tz = Tz::Asia__Bahrain;
    pub const Baku: Tz = Tz::Asia__Baku;
    pub const Bangkok: Tz = Tz::Asia__Bangkok;
    pub const Barnaul: Tz = Tz::Asia__Barnaul;
    pub const Beirut: Tz = Tz::Asia__Beirut;
    pub const Bishkek: Tz = Tz::Asia__Bishkek;
    pub const Brunei: Tz = Tz::Asia__Brunei;
    pub const Calcutta: Tz = Tz::Asia__Calcutta;
    pub const Chita: Tz = Tz::Asia__Chita;
    pub const Choibalsan: Tz = Tz::Asia__Choibalsan;
    pub const Chongqing: Tz = Tz::Asia__Chongqing;
    pub const Chungking: Tz = Tz::Asia__Chungking;
    pub const Colombo: Tz = Tz::Asia__Colombo;
    pub const Dacca: Tz = Tz::Asia__Dacca;
    pub const Damascus: Tz = Tz::Asia__Damascus;
    pub const Dhaka: Tz = Tz::Asia__Dhaka;
    pub const Dili: Tz = Tz::Asia__Dili;
    pub const Dubai: Tz = Tz::Asia__Dubai;
    pub const Dushanbe: Tz = Tz::Asia__Dushanbe;
    pub const Famagusta: Tz = Tz::Asia__Famagusta;
    pub const Gaza: Tz = Tz::Asia__Gaza;
    pub const Harbin: Tz = Tz::Asia__Harbin;
    pub const Hebron: Tz = Tz::Asia__Hebron;
    pub const Ho_Chi_Minh: Tz = Tz::Asia__Ho_Chi_Minh;
    pub const Hong_Kong: Tz = Tz::Asia__Hong_Kong;
    pub const Hovd: Tz = Tz::Asia__Hovd;
    pub const Irkutsk: Tz = Tz::Asia__Irkutsk;
    pub const Istanbul: Tz = Tz::Asia__Istanbul;
    pub const Jakarta: Tz = Tz::Asia__Jakarta;
    pub const Jayapura: Tz = Tz::Asia__Jayapura;
    pub const Jerusalem: Tz = Tz::Asia__Jerusalem;
    pub const Kabul: Tz = Tz::Asia__Kabul;
    pub const Kamchatka: Tz = Tz::Asia__Kamchatka;
    pub const Karachi: Tz = Tz::Asia__Karachi;
    pub const Kashgar: Tz = Tz::Asia__Kashgar;
    pub const Kathmandu: Tz = Tz::Asia__Kathmandu;
    pub const Katmandu: Tz = Tz::Asia__Katmandu;
    pub const Khandyga: Tz = Tz::Asia__Khandyga;
    pub const Kolkata: Tz = Tz::Asia__Kolkata;
    pub const Krasnoyarsk: Tz = Tz::Asia__Krasnoyarsk;
    pub const Kuala_Lumpur: Tz = Tz::Asia__Kuala_Lumpur;
    pub const Kuching: Tz = Tz::Asia__Kuching;
    pub const Kuwait: Tz = Tz::Asia__Kuwait;
    pub const Macao: Tz = Tz::Asia__Macao;
    pub const Macau: Tz = Tz::Asia__Macau;
    pub const Magadan: Tz = Tz::Asia__Magadan;
    pub const Makassar: Tz = Tz::Asia__Makassar;
    pub const Manila: Tz = Tz::Asia__Manila;
    pub const Muscat: Tz = Tz::Asia__Muscat;
    pub const Nicosia: Tz = Tz::Asia__Nicosia;
    pub const Novokuznetsk: Tz = Tz::Asia__Novokuznetsk;
    pub const Novosibirsk: Tz = Tz::Asia__Novosibirsk;
    pub const Omsk: Tz = Tz::Asia__Omsk;
    pub const Oral: Tz = Tz::Asia__Oral;
    pub const Phnom_Penh: Tz = Tz::Asia__Phnom_Penh;
    pub const Pontianak: Tz = Tz::Asia__Pontianak;
    pub const Pyongyang: Tz = Tz::Asia__Pyongyang;
    pub const Qatar: Tz = Tz::Asia__Qatar;
    pub const Qostanay: Tz = Tz::Asia__Qostanay;
    pub const Qyzylorda: Tz = Tz::Asia__Qyzylorda;
    pub const Rangoon: Tz = Tz::Asia__Rangoon;
    pub const Riyadh: Tz = Tz::Asia__Riyadh;
    pub const Saigon: Tz = Tz::Asia__Saigon;
    pub const Sakhalin: Tz = Tz::Asia__Sakhalin;
    pub const Samarkand: Tz = Tz::Asia__Samarkand;
    pub const Seoul: Tz = Tz::Asia__Seoul;
    pub const Shanghai: Tz = Tz::Asia__Shanghai;
    pub const Singapore: Tz = Tz::Asia__Singapore;
    pub const Srednekolymsk: Tz = Tz::Asia__Srednekolymsk;
    pub const Taipei: Tz = Tz::Asia__Taipei;
    pub const Tashkent: Tz = Tz::Asia__Tashkent;
    pub const Tbilisi: Tz = Tz::Asia__Tbilisi;
    pub const Tehran: Tz = Tz::Asia__Tehran;
    pub const Tel_Aviv: Tz = Tz::Asia__Tel_Aviv;
    pub const Thimbu: Tz = Tz::Asia__Thimbu;
    pub const Thimphu: Tz = Tz::Asia__Thimphu;
    pub const Tokyo: Tz = Tz::Asia__Tokyo;
    pub const Tomsk: Tz = Tz::Asia__Tomsk;
    pub const Ujung_Pandang: Tz = Tz::Asia__Ujung_Pandang;
    pub const Ulaanbaatar: Tz = Tz::Asia__Ulaanbaatar;
    pub const Ulan_Bator: Tz = Tz::Asia__Ulan_Bator;
    pub const Urumqi: Tz = Tz::Asia__Urumqi;
    pub const UstNera: Tz = Tz::Asia__UstNera;
    pub const Vientiane: Tz = Tz::Asia__Vientiane;
    pub const Vladivostok: Tz = Tz::Asia__Vladivostok;
    pub const Yakutsk: Tz = Tz::Asia__Yakutsk;
    pub const Yangon: Tz = Tz::Asia__Yangon;
    pub const Yekaterinburg: Tz = Tz::Asia__Yekaterinburg;
    pub const Yerevan: Tz = Tz::Asia__Yerevan;
}

pub mod Atlantic {
    use crate::timezones::Tz;

    pub const Azores: Tz = Tz::Atlantic__Azores;
    pub const Bermuda: Tz = Tz::Atlantic__Bermuda;
    pub const Canary: Tz = Tz::Atlantic__Canary;
    pub const Cape_Verde: Tz = Tz::Atlantic__Cape_Verde;
    pub const Faeroe: Tz = Tz::Atlantic__Faeroe;
    pub const Faroe: Tz = Tz::Atlantic__Faroe;
    pub const Jan_Mayen: Tz = Tz::Atlantic__Jan_Mayen;
    pub const Madeira: Tz = Tz::Atlantic__Madeira;
    pub const Reykjavik: Tz = Tz::Atlantic__Reykjavik;
    pub const South_Georgia: Tz = Tz::Atlantic__South_Georgia;
    pub const St_Helena: Tz = Tz::Atlantic__St_Helena;
    pub const Stanley: Tz = Tz::Atlantic__Stanley;
}

pub mod Australia {
    use crate::timezones::Tz;

    pub const ACT: Tz = Tz::Australia__ACT;
    pub const Adelaide: Tz = Tz::Australia__Adelaide;
    pub const Brisbane: Tz = Tz::Australia__Brisbane;
    pub const Broken_Hill: Tz = Tz::Australia__Broken_Hill;
    pub const Canberra: Tz = Tz::Australia__Canberra;
    pub const Currie: Tz = Tz::Australia__Currie;
    pub const Darwin: Tz = Tz::Australia__Darwin;
    pub const Eucla: Tz = Tz::Australia__Eucla;
    pub const Hobart: Tz = Tz::Australia__Hobart;
    pub const LHI: Tz = Tz::Australia__LHI;
    pub const Lindeman: Tz = Tz::Australia__Lindeman;
    pub const Lord_Howe: Tz = Tz::Australia__Lord_Howe;
    pub const Melbourne: Tz = Tz::Australia__Melbourne;
    pub const NSW: Tz = Tz::Australia__NSW;
    pub const North: Tz = Tz::Australia__North;
    pub const Perth: Tz = Tz::Australia__Perth;
    pub const Queensland: Tz = Tz::Australia__Queensland;
    pub const South: Tz = Tz::Australia__South;
    pub const Sydney: Tz = Tz::Australia__Sydney;
    pub const Tasmania: Tz = Tz::Australia__Tasmania;
    pub const Victoria: Tz = Tz::Australia__Victoria;
    pub const West: Tz = Tz::Australia__West;
    pub const Yancowinna: Tz = Tz::Australia__Yancowinna;
}

pub mod Brazil {
    use crate::timezones::Tz;

    pub const Acre: Tz = Tz::Brazil__Acre;
    pub const DeNoronha: Tz = Tz::Brazil__DeNoronha;
    pub const East: Tz = Tz::Brazil__East;
    pub const West: Tz = Tz::Brazil__West;
}

pub mod Canada {
    use crate::timezones::Tz;

    pub const Atlantic: Tz = Tz::Canada__Atlantic;
    pub const Central: Tz = Tz::Canada__Central;
    pub const Eastern: Tz = Tz::Canada__Eastern;
    pub const Mountain: Tz = Tz::Canada__Mountain;
    pub const Newfoundland: Tz = Tz::Canada__Newfoundland;
    pub const Pacific: Tz = Tz::Canada__Pacific;
    pub const Saskatchewan: Tz = Tz::Canada__Saskatchewan;
    pub const Yukon: Tz = Tz::Canada__Yukon;
}

pub mod Chile {
    use crate::timezones::Tz;

    pub const Continental: Tz = Tz::Chile__Continental;
    pub const EasterIsland: Tz = Tz::Chile__EasterIsland;
}

pub mod Etc {
    use crate::timezones::Tz;

    pub const GMT: Tz = Tz::Etc__GMT;
    pub const GMTPlus0: Tz = Tz::Etc__GMTPlus0;
    pub const GMTPlus1: Tz = Tz::Etc__GMTPlus1;
    pub const GMTPlus10: Tz = Tz::Etc__GMTPlus10;
    pub const GMTPlus11: Tz = Tz::Etc__GMTPlus11;
    pub const GMTPlus12: Tz = Tz::Etc__GMTPlus12;
    pub const GMTPlus2: Tz = Tz::Etc__GMTPlus2;
    pub const GMTPlus3: Tz = Tz::Etc__GMTPlus3;
    pub const GMTPlus4: Tz = Tz::Etc__GMTPlus4;
    pub const GMTPlus5: Tz = Tz::Etc__GMTPlus5;
    pub const GMTPlus6: Tz = Tz::Etc__GMTPlus6;
    pub const GMTPlus7: Tz = Tz::Etc__GMTPlus7;
    pub const GMTPlus8: Tz = Tz::Etc__GMTPlus8;
    pub const GMTPlus9: Tz = Tz::Etc__GMTPlus9;
    pub const GMTMinus0: Tz = Tz::Etc__GMTMinus0;
    pub const GMTMinus1: Tz = Tz::Etc__GMTMinus1;
    pub const GMTMinus10: Tz = Tz::Etc__GMTMinus10;
    pub const GMTMinus11: Tz = Tz::Etc__GMTMinus11;
    pub const GMTMinus12: Tz = Tz::Etc__GMTMinus12;
    pub const GMTMinus13: Tz = Tz::Etc__GMTMinus13;
    pub const GMTMinus14: Tz = Tz::Etc__GMTMinus14;
    pub const GMTMinus2: Tz = Tz::Etc__GMTMinus2;
    pub const GMTMinus3: Tz = Tz::Etc__GMTMinus3;
    pub const GMTMinus4: Tz = Tz::Etc__GMTMinus4;
    pub const GMTMinus5: Tz = Tz::Etc__GMTMinus5;
    pub const GMTMinus6: Tz = Tz::Etc__GMTMinus6;
    pub const GMTMinus7: Tz = Tz::Etc__GMTMinus7;
    pub const GMTMinus8: Tz = Tz::Etc__GMTMinus8;
    pub const GMTMinus9: Tz = Tz::Etc__GMTMinus9;
    pub const GMT0: Tz = Tz::Etc__GMT0;
    pub const Greenwich: Tz = Tz::Etc__Greenwich;
    pub const UCT: Tz = Tz::Etc__UCT;
    pub const UTC: Tz = Tz::Etc__UTC;
    pub const Universal: Tz = Tz::Etc__Universal;
    pub const Zulu: Tz = Tz::Etc__Zulu;
}

pub mod Europe {
    use crate::timezones::Tz;

    pub const Amsterdam: Tz = Tz::Europe__Amsterdam;
    pub const Andorra: Tz = Tz::Europe__Andorra;
    pub const Astrakhan: Tz = Tz::Europe__Astrakhan;
    pub const Athens: Tz = Tz::Europe__Athens;
    pub const Belfast: Tz = Tz::Europe__Belfast;
    pub const Belgrade: Tz = Tz::Europe__Belgrade;
    pub const Berlin: Tz = Tz::Europe__Berlin;
    pub const Bratislava: Tz = Tz::Europe__Bratislava;
    pub const Brussels: Tz = Tz::Europe__Brussels;
    pub const Bucharest: Tz = Tz::Europe__Bucharest;
    pub const Budapest: Tz = Tz::Europe__Budapest;
    pub const Busingen: Tz = Tz::Europe__Busingen;
    pub const Chisinau: Tz = Tz::Europe__Chisinau;
    pub const Copenhagen: Tz = Tz::Europe__Copenhagen;
    pub const Dublin: Tz = Tz::Europe__Dublin;
    pub const Gibraltar: Tz = Tz::Europe__Gibraltar;
    pub const Guernsey: Tz = Tz::Europe__Guernsey;
    pub const Helsinki: Tz = Tz::Europe__Helsinki;
    pub const Isle_of_Man: Tz = Tz::Europe__Isle_of_Man;
    pub const Istanbul: Tz = Tz::Europe__Istanbul;
    pub const Jersey: Tz = Tz::Europe__Jersey;
    pub const Kaliningrad: Tz = Tz::Europe__Kaliningrad;
    pub const Kiev: Tz = Tz::Europe__Kiev;
    pub const Kirov: Tz = Tz::Europe__Kirov;
    pub const Kyiv: Tz = Tz::Europe__Kyiv;
    pub const Lisbon: Tz = Tz::Europe__Lisbon;
    pub const Ljubljana: Tz = Tz::Europe__Ljubljana;
    pub const London: Tz = Tz::Europe__London;
    pub const Luxembourg: Tz = Tz::Europe__Luxembourg;
    pub const Madrid: Tz = Tz::Europe__Madrid;
    pub const Malta: Tz = Tz::Europe__Malta;
    pub const Mariehamn: Tz = Tz::Europe__Mariehamn;
    pub const Minsk: Tz = Tz::Europe__Minsk;
    pub const Monaco: Tz = Tz::Europe__Monaco;
    pub const Moscow: Tz = Tz::Europe__Moscow;
    pub const Nicosia: Tz = Tz::Europe__Nicosia;
    pub const Oslo: Tz = Tz::Europe__Oslo;
    pub const Paris: Tz = Tz::Europe__Paris;
    pub const Podgorica: Tz = Tz::Europe__Podgorica;
    pub const Prague: Tz = Tz::Europe__Prague;
    pub const Riga: Tz = Tz::Europe__Riga;
    pub const Rome: Tz = Tz::Europe__Rome;
    pub const Samara: Tz = Tz::Europe__Samara;
    pub const San_Marino: Tz = Tz::Europe__San_Marino;
    pub const Sarajevo: Tz = Tz::Europe__Sarajevo;
    pub const Saratov: Tz = Tz::Europe__Saratov;
    pub const Simferopol: Tz = Tz::Europe__Simferopol;
    pub const Skopje: Tz = Tz::Europe__Skopje;
    pub const Sofia: Tz = Tz::Europe__Sofia;
    pub const Stockholm: Tz = Tz::Europe__Stockholm;
    pub const Tallinn: Tz = Tz::Europe__Tallinn;
    pub const Tirane: Tz = Tz::Europe__Tirane;
    pub const Tiraspol: Tz = Tz::Europe__Tiraspol;
    pub const Ulyanovsk: Tz = Tz::Europe__Ulyanovsk;
    pub const Uzhgorod: Tz = Tz::Europe__Uzhgorod;
    pub const Vaduz: Tz = Tz::Europe__Vaduz;
    pub const Vatican: Tz = Tz::Europe__Vatican;
    pub const Vienna: Tz = Tz::Europe__Vienna;
    pub const Vilnius: Tz = Tz::Europe__Vilnius;
    pub const Volgograd: Tz = Tz::Europe__Volgograd;
    pub const Warsaw: Tz = Tz::Europe__Warsaw;
    pub const Zagreb: Tz = Tz::Europe__Zagreb;
    pub const Zaporozhye: Tz = Tz::Europe__Zaporozhye;
    pub const Zurich: Tz = Tz::Europe__Zurich;
}

pub mod Indian {
    use crate::timezones::Tz;

    pub const Antananarivo: Tz = Tz::Indian__Antananarivo;
    pub const Chagos: Tz = Tz::Indian__Chagos;
    pub const Christmas: Tz = Tz::Indian__Christmas;
    pub const Cocos: Tz = Tz::Indian__Cocos;
    pub const Comoro: Tz = Tz::Indian__Comoro;
    pub const Kerguelen: Tz = Tz::Indian__Kerguelen;
    pub const Mahe: Tz = Tz::Indian__Mahe;
    pub const Maldives: Tz = Tz::Indian__Maldives;
    pub const Mauritius: Tz = Tz::Indian__Mauritius;
    pub const Mayotte: Tz = Tz::Indian__Mayotte;
    pub const Reunion: Tz = Tz::Indian__Reunion;
}

pub mod Mexico {
    use crate::timezones::Tz;

    pub const BajaNorte: Tz = Tz::Mexico__BajaNorte;
    pub const BajaSur: Tz = Tz::Mexico__BajaSur;
    pub const General: Tz = Tz::Mexico__General;
}

pub mod Pacific {
    use crate::timezones::Tz;

    pub const Apia: Tz = Tz::Pacific__Apia;
    pub const Auckland: Tz = Tz::Pacific__Auckland;
    pub const Bougainville: Tz = Tz::Pacific__Bougainville;
    pub const Chatham: Tz = Tz::Pacific__Chatham;
    pub const Chuuk: Tz = Tz::Pacific__Chuuk;
    pub const Easter: Tz = Tz::Pacific__Easter;
    pub const Efate: Tz = Tz::Pacific__Efate;
    pub const Enderbury: Tz = Tz::Pacific__Enderbury;
    pub const Fakaofo: Tz = Tz::Pacific__Fakaofo;
    pub const Fiji: Tz = Tz::Pacific__Fiji;
    pub const Funafuti: Tz = Tz::Pacific__Funafuti;
    pub const Galapagos: Tz = Tz::Pacific__Galapagos;
    pub const Gambier: Tz = Tz::Pacific__Gambier;
    pub const Guadalcanal: Tz = Tz::Pacific__Guadalcanal;
    pub const Guam: Tz = Tz::Pacific__Guam;
    pub const Honolulu: Tz = Tz::Pacific__Honolulu;
    pub const Johnston: Tz = Tz::Pacific__Johnston;
    pub const Kanton: Tz = Tz::Pacific__Kanton;
    pub const Kiritimati: Tz = Tz::Pacific__Kiritimati;
    pub const Kosrae: Tz = Tz::Pacific__Kosrae;
    pub const Kwajalein: Tz = Tz::Pacific__Kwajalein;
    pub const Majuro: Tz = Tz::Pacific__Majuro;
    pub const Marquesas: Tz = Tz::Pacific__Marquesas;
    pub const Midway: Tz = Tz::Pacific__Midway;
    pub const Nauru: Tz = Tz::Pacific__Nauru;
    pub const Niue: Tz = Tz::Pacific__Niue;
    pub const Norfolk: Tz = Tz::Pacific__Norfolk;
    pub const Noumea: Tz = Tz::Pacific__Noumea;
    pub const Pago_Pago: Tz = Tz::Pacific__Pago_Pago;
    pub const Palau: Tz = Tz::Pacific__Palau;
    pub const Pitcairn: Tz = Tz::Pacific__Pitcairn;
    pub const Pohnpei: Tz = Tz::Pacific__Pohnpei;
    pub const Ponape: Tz = Tz::Pacific__Ponape;
    pub const Port_Moresby: Tz = Tz::Pacific__Port_Moresby;
    pub const Rarotonga: Tz = Tz::Pacific__Rarotonga;
    pub const Saipan: Tz = Tz::Pacific__Saipan;
    pub const Samoa: Tz = Tz::Pacific__Samoa;
    pub const Tahiti: Tz = Tz::Pacific__Tahiti;
    pub const Tarawa: Tz = Tz::Pacific__Tarawa;
    pub const Tongatapu: Tz = Tz::Pacific__Tongatapu;
    pub const Truk: Tz = Tz::Pacific__Truk;
    pub const Wake: Tz = Tz::Pacific__Wake;
    pub const Wallis: Tz = Tz::Pacific__Wallis;
    pub const Yap: Tz = Tz::Pacific__Yap;
}

pub mod US {
    use crate::timezones::Tz;

    pub const Alaska: Tz = Tz::US__Alaska;
    pub const Aleutian: Tz = Tz::US__Aleutian;
    pub const Arizona: Tz = Tz::US__Arizona;
    pub const Central: Tz = Tz::US__Central;
    pub const EastIndiana: Tz = Tz::US__EastIndiana;
    pub const Eastern: Tz = Tz::US__Eastern;
    pub const Hawaii: Tz = Tz::US__Hawaii;
    pub const IndianaStarke: Tz = Tz::US__IndianaStarke;
    pub const Michigan: Tz = Tz::US__Michigan;
    pub const Mountain: Tz = Tz::US__Mountain;
    pub const Pacific: Tz = Tz::US__Pacific;
    pub const Samoa: Tz = Tz::US__Samoa;
}
